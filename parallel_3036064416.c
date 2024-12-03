/*
* PLEASE WRITE DOWN FOLLOWING INFO BEFORE SUBMISSION
* FILE NAME: parallel_3036064416.c
* NAME: Ma Tin Yu
* UID:  3036064416
* Development Platform: Vscode (workbench2)
* Remark: The implementations, i.e. especially the design of lines 170-194 is inspired from two youtube videos: https://  www.youtube.com/watch?v=_n2hE2gyPxU&t=921s and https://www.youtube.com/watch?v=7i9z4CRYLAE&t=374s&ab_channel=CodeVault . These two videos help me a lot when implementing thread pools on taking input from the api functions and then store them into an array with first-in-first-out feature that the thread will take the left-most work and then do something related to the work, which is stored in an object consisting of the method and instant variables.
The implementation use mutex and conditional variable because I cannot handle the semaphore correctly.
* How to compile separately: (gcc -o parallel parallel_3036064416.c -O2 -lm -lpthread)
*/

#include "common.h" // some common definitions

#include <unistd.h>       // for nearly everything :)
#include <stdio.h>        // for printf, sprintf, fgets
#include <stdlib.h>       // for malloc, calloc
#include <stdint.h>       // for uint8_t and uint64_t
#include <time.h>         // for time
#include <string.h>       // for memcpy and strcmp
#include <sys/resource.h> // for rusage collection

#include "model.h"// for Llama definitions -> no need to know

int pos = 0; // global position of generation
Transformer transformer; // transformer instance to be init
Tokenizer tokenizer;     // tokenizer instance to be init
Sampler sampler;         // sampler instance to be init

// YOUR CODE STARTS HERE
#include <pthread.h>
// #include <semaphore.h> // uncomment this line if you use semaphore
#include <stdbool.h>   // uncomment this line if you want true / false

// you may define global variables here

int threadCount;    // number of thread shared globally
int workCountMat = 0;   // number of works for mat-vec-mult
int workCountMult = 0;  // number of works for mult-head
int workCountTotal = 0; // total number of work in the whole process 
pthread_cond_t condForMainMat, condForMainMult,condForThreads;  // conditional variable for signal from threads handling mat-vec-mult to main, threads handling multi-head to main and main to all threads respectively
pthread_mutex_t lock;   // just a mutex lock
bool isEnd = false;     // check for end of process

typedef struct{     // struct for storing thread usage
    double utime;
    double stime;
}Threadusage;

typedef struct{     // struct for parameters in mat-vec-mult function
    int id;
    float* out;
    QuantizedTensor *vec;
    QuantizedTensor *mat;
    int col;
    int row;
    int start;
    int end;
    int handle;
}MatvecmultPara;

typedef struct{     //struct for parameters in multi-head function
    int id;
    float* out;
    int start;
    int end;
    int handle;
    float* q;          
    float* key_cache;  
    float* value_cache;
    float* att;        
    int seq_len;
    int n_heads;
    int head_size;
    int kv_dim;
    int kv_mul;
}MultiheadPara;

typedef struct{     // struct for works allocation
    void (*workFunc)(int,void*);
    void* para;
    int workId;
}WorkLoad;

pthread_t *threads; // array storing threads
WorkLoad *worklist; // array storing the works
Threadusage *threadusage;   // array storing usage of threads in order of thread id

// function executed by each thread to complete mat_vec_mul
// @note: please modify the signature to what you want
void mat_vec_mul_task_func(int id, void* para) {
    MatvecmultPara *paras = (MatvecmultPara *) para;    // cast the MatvecmultPara pointer type to the void pointer argument
    for (int i = paras->start; i < paras->end; i++) {
        float val = 0.0f; // final value
        int32_t ival = 0; // integer value to be dequantized
        int in = i * paras->col;   // 

        // for each column
        // GS is group size of quantization, not included in assignment
        // @note please don't parallel this loop
        for (int j = 0; j <= paras->col - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) paras->vec->q[j + k]) * ((int32_t) paras->mat->q[in + j + k]);
            }
            val += ((float) ival) * paras->mat->s[(in + j) / GS] * paras->vec->s[j / GS];
            ival = 0;
        }
        paras->out[i] = val;
    }
    free(paras);    // release the memory for data pointer
}

// function executed by each thread to complete multi_head_attn
// @note: please modify the signature to what you want
void multi_head_attn_task_func(int id,void* para) {
    MultiheadPara *paras = (MultiheadPara *) para;  // cast the MultiheadPara pointer type to the void pointer argument
    for (int h = paras->start; h < paras->end; h++) {
        // get the query vector for this head
        float* head_q = paras->q + h * paras->head_size;
        // attention scores for this head
        float* head_att = paras->att + h * paras->seq_len;
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            float* head_k = paras->key_cache + t * paras->kv_dim + (h / paras->kv_mul) * paras->head_size;
            // calculate the attention score as the dot product of q and k
            float score = 0.0f;
            for (int i = 0; i < paras->head_size; i++) {
                score += head_q[i] * head_k[i];
            }
            score /= sqrtf(paras->head_size);
            // save the score to the attention buffer
            head_att[t] = score;
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax(head_att, pos + 1);

        // weighted sum of the values, store back into xb
        float* head_out = paras->out + h * paras->head_size;
        memset(head_out, 0, paras->head_size * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            float* head_v = paras->value_cache + t * paras->kv_dim + (h / paras->kv_mul) * paras->head_size;
            // get the attention weight for this timestep
            float a = head_att[t];
            // accumulate the weighted value into head out
            for (int i = 0; i < paras->head_size; i++) {
                head_out[i] += a * head_v[i];
            }
        }
    }   
    free(paras);    // release the memory for data pointer
}

// thread function used in pthread_create
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void *thr_func(void *arg) {
    int id  = *(int*) arg;  // cast integer type to void argument pointer
    while(1){
        WorkLoad work;  // initialize a work
        pthread_mutex_lock(&lock);
        while(workCountTotal==0 && !isEnd){     // wait for main collecting data
            pthread_cond_wait(&condForThreads,&lock);
        }
        if(isEnd){  // getting signal from close_thr_pool and terminate the thr_func
            pthread_mutex_unlock(&lock);
            struct rusage usage;
            int ret;
            ret = getrusage(RUSAGE_THREAD, &usage); // get usage
            if(ret != 0){
                perror("Error in get system usage");
            }
            Threadusage thread; // initialize the usage struct
            thread.utime = (double)usage.ru_utime.tv_sec + (double)usage.ru_utime.tv_usec / (double)1e6;
            thread.stime = (double)usage.ru_stime.tv_sec + (double)usage.ru_stime.tv_usec / (double)1e6;
            pthread_mutex_lock(&lock);
            threadusage[id] = thread;   // store usage of Thread [id] to the array
            pthread_mutex_unlock(&lock);
            break;
        }
        // pop the leftmost work and pass it to the thread to handle, the logic is using a queue-like structure to take the item out with first-in-first-out feature. The idea is inspired from https://www.youtube.com/watch?v=_n2hE2gyPxU&t=921s&ab_channel=CodeVault starting at 9:38
        work = worklist[0];
        workCountTotal--;
        // shift the items to the left for upcoming pop
        for(int i = 0; i<workCountTotal;i++){
            worklist[i] = worklist[i+1];
        }
        pthread_mutex_unlock(&lock);
        if(work.workId == 0){   // Do mat-vec-mult
            work.workFunc(id,work.para);    // function call using functino pointer
            pthread_mutex_lock(&lock);
            workCountMat--;     // decrease number of work after finishing the job
            if(workCountMat == 0){
                pthread_cond_signal(&condForMainMat);   // signal main thread that all works for mat-vec-mult are done
            }
            pthread_mutex_unlock(&lock);
        }
        else{   // Do mult-head
            work.workFunc(id,work.para);    // function call using functino pointer
            pthread_mutex_lock(&lock);
            workCountMult--;    // decrease number of work after finishing the job
            if(workCountMult == 0){
                pthread_cond_signal(&condForMainMult);  // signal main thread that all works for mult-head are done
            }
            pthread_mutex_unlock(&lock);

        }

    }
}

// function to initialize thread pool
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void init_thr_pool(int num_thr) {
    threadCount = num_thr;  // copying number of thread for global use
    threads = malloc(threadCount * sizeof(pthread_t));  // initialize array for threads
    threadusage = malloc(threadCount * sizeof(Threadusage));    // initialize array for thread usage
    worklist = malloc(1000 * sizeof(WorkLoad)); // initialize array for works
    pthread_cond_init(&condForMainMat,NULL);    // initialize conditional variables
    pthread_cond_init(&condForMainMult,NULL);
    pthread_cond_init(&condForThreads,NULL);
    pthread_mutex_init(&lock,NULL); // initialize array for mutex
    for(int i = 0; i<threadCount; i++){
        int *id = malloc(sizeof(int));  
        *id = i;    // get and allocate id for threads
        pthread_create(&threads[i],NULL,thr_func,id);
    }
}

// function to close thread pool
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void close_thr_pool() {
    isEnd = true;   // inform all threads to stop and terminate vai global shared variable
    for(int i = 0;i<threadCount;i++){
        pthread_cond_signal(&condForThreads);   // wake up all threads to collect usage
    }
    usleep(1000);   // wait for collection
    for(int i = 0;i<threadCount;i++){
        pthread_join(threads[i],NULL);  // terminate all the threads
        printf("Thread %d has completed - user: %.4f s, system: %.4f s\n",i,threadusage[i].utime,threadusage[i].stime);
    }   // print usage for each threads in ascending order
    int ret;
    struct rusage usageMain;
    ret = getrusage(RUSAGE_THREAD, &usageMain);    // get usage of main thread
    if(ret != 0){
        printf("Error in get system usage");
    }
    printf("Main thread has completed - user: %.4f s, system: %.4f s\n",(double)usageMain.ru_utime.tv_sec + (double)usageMain.ru_utime.tv_usec / (double)1e6,(double)usageMain.ru_stime.tv_sec + (double)usageMain.ru_stime.tv_usec / (double)1e6);
    struct rusage usageWhole;   // get usage of the whole process
    ret = getrusage(RUSAGE_SELF, &usageWhole);
    if(ret != 0){
        printf("Error in get system usage");
    }
    double totalUserTime = (double)usageWhole.ru_utime.tv_sec + (double)usageWhole.ru_utime.tv_usec / (double)1e6;
    double totalSysTime = (double)usageWhole.ru_stime.tv_sec + (double)usageWhole.ru_stime.tv_usec / (double)1e6;
    printf("Whole process - user: %.4f s, system: %.4f s\n",totalUserTime, totalSysTime);
    pthread_mutex_destroy(&lock);   // destory mutex
    pthread_cond_destroy(&condForMainMat);  // destory conditional variables
    pthread_cond_destroy(&condForMainMult);
    pthread_cond_destroy(&condForThreads);
    free(threads);  // release memory
    free(threadusage);
    free(worklist);
}

// ----------------------------------------------------------------------------
// entry function for multi-threading matrix multiplication
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void mat_vec_mul(float* out, QuantizedTensor *vec, QuantizedTensor *mat, int col, int row) {
    int handle;
    handle = row/threadCount;   // get number of row handle by each thread
    if(handle * threadCount != row){    // check if number of thread divided row
        handle++;   // round up if not divisble
    }
    for(int i = 0 ; i< threadCount;i++){
        MatvecmultPara *para = malloc(sizeof(MatvecmultPara));  // initialize the data pointers
        para->id = i;
        para->handle = handle;
        para->out = out;
        para->vec = vec;
        para->mat = mat;
        para->row = row;
        para->col = col;
        para->start = i * handle;
        if((i == threadCount - 1) && (handle * threadCount != row)){
            para->end = row ;   // set end point for the last thread if not divisble 
        }else{
            para->end = i * handle + handle;    // set normal end point for those (n-1) threads
        }
        WorkLoad work ;
        work.workFunc = &mat_vec_mul_task_func;
        work.para = para;
        work.workId = 0;
        pthread_mutex_lock(&lock);
        worklist[workCountTotal] = work;    // store work in the worklist and wait for thread handle it
        workCountMat++; // update number of work for mat-vec-mult
        workCountTotal++;   // update total number of work
        pthread_mutex_unlock(&lock);
    }
    for(int i =0;i<threadCount;i++){
        pthread_cond_signal(&condForThreads);   // wake up all the threads for work
    }
    pthread_mutex_lock(&lock);
    while(workCountMat != 0){
        pthread_cond_wait(&condForMainMat,&lock);   // wait for work done
    }
    pthread_mutex_unlock(&lock);
}

// ----------------------------------------------------------------------------
// entry function for multi-threading multi-head-attention
// @note: YOU CAN NOT MODIFY FUNCTION SIGNATURE!!!
void multi_head_attn(
    float* out,         // output tensor [head, head_size]
    float* q,           // query tensor  [head, head_size]
    float* key_cache,   // cache of history key tensor   [kv_head, seq_len, head_size]
    float* value_cache, // cache of history value tensor [kv_head, seq_len, head_size]
    float* att,         // buffer for attention score [head, seq_len]
    int seq_len,
    int n_heads,
    int head_size,
    int kv_dim,
    int kv_mul) {
    int handle;
    int numOfHead = n_heads;
    int stopThread =99999;
    handle = numOfHead / threadCount;   // get number of head handle by each thread
    if(handle * threadCount != numOfHead){  // check if number of thread divided number of heads
        handle++;   // round up if not divisble
        for(int j = 0 ; j < threadCount;j++){   // find the last thread that will handle the last head since not all the thread will be used in mult-head
            if(numOfHead < handle){
                stopThread = j; // locate the stopping thread
                break;
            }
            numOfHead -= handle;
        }
    }
    for(int i = 0; i< threadCount;i++){
         MultiheadPara* para = malloc(sizeof(MultiheadPara));   // initialize the data pointer
        if(i < stopThread){
            para->end = handle + handle * i;    // set normal end point for those (stopThread -1) threads
        }else if(i == stopThread){  // set the end point for the stopping thread the last head 
            para->end = n_heads;
        }else{
            free(para); // skip those useless threads 
            continue;
        }
        para->id = i;
        para->out = out;
        para->start = i * handle;
        para->handle = handle;
        para->q = q;
        para->key_cache = key_cache;
        para->value_cache = value_cache;
        para->att = att;
        para->seq_len =seq_len;
        para->n_heads = n_heads;
        para->head_size = head_size;
        para->kv_dim = kv_dim;
        para->kv_mul = kv_mul;
        WorkLoad work;
        work.workFunc = &multi_head_attn_task_func ;
        work.para = para;
        work.workId = 1;
        pthread_mutex_lock(&lock);
        worklist[workCountTotal] = work;    // store work in the worklist and wait for thread handle it
        workCountMult++;     // update number of work for mult-head
        workCountTotal++;   // update total number of work
        pthread_mutex_unlock(&lock);
    }
    for(int i =0;i<threadCount;i++){
        pthread_cond_signal(&condForThreads); // wake up all the threads for work
    }
    pthread_mutex_lock(&lock);
    while(workCountMult != 0){
        pthread_cond_wait(&condForMainMult,&lock);  // wait for work done
    }
    pthread_mutex_unlock(&lock);
}
// YOUR CODE ENDS HERE

// ----------------------------------------------------------------------------
// forward Transformer, you're not allowed to modify this part
float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->q, &s->xq, w->wq + l, dim, dim);
        mat_vec_mul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        mat_vec_mul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        multi_head_attn(s->xb, s->q, s->key_cache + loff, s->value_cache + loff, s->att, 
            p->seq_len, p->n_heads, head_size, kv_dim, kv_mul);

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->xb2, &s->xq, w->wo + l, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        mat_vec_mul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, hidden_dim);
        mat_vec_mul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    mat_vec_mul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// generation loop, you're not allowed to modify this part
void generate(char *prompt) {
    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+6) * sizeof(int)); // +6 reserved for prompt template
    encode(&tokenizer, prompt, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    int next;        // place holder for next token
    int token = prompt_tokens[0]; // place holder of prev token, kickoff as prompt_tokens[0]
    int end_pos = pos + MAX_NEW_TOKENS + num_prompt_tokens;
    int start_pos = pos;
    long start_time = 0; // to be lazy iniialzied
    while (pos < end_pos) {

        // forward the transformer to get logits for the next token
        float* logits = forward(&transformer, token, pos);

        if (pos < start_pos + num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos - start_pos + 1];
        } else if (pos == end_pos - 2) {
            // reaching the end, force it to close by <|im_end|>
            next = 2; // := <|im_end|>
        } else {
            // otherwise sample the next token from the logits
            next = sample(&sampler, logits);
        }

        pos++;

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(&tokenizer, token, next);
        if (pos >= num_prompt_tokens) {
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }

        token = next;

        // init the timer here because the first iteration can be slower
        if (start_time == 0) { start_time = time_in_ms(); }
    }
    printf("\n");

    long end_time = time_in_ms();
    // \033[0;32m set color to green and \033[0m reset to default, they won't be generate by LLM
    fprintf(stdout, "\033[0;32mlength: %d, speed (tok/s): %.4f \033[0m\n", 
        pos, (pos - start_pos) / (float) (end_time - start_time) * 1000);
    
    free(prompt_tokens);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *model_path     = "model.bin";  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature    = 0.6f;  // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp           = 0.9f;  // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    char *prompt         = NULL;  // prompt strings
    int num_prompt       = 0; // number of prompts
    uint64_t rng_seed    = 0; // seed rng with time by default
    int num_thr          = 0;

    if (argc == 4) {
        num_thr  = atoi(argv[1]);
        rng_seed = atoi(argv[2]);
        prompt   = argv[3];
    } else {
        fprintf(stderr, "Usage:   ./seq <num_thr> <seed> <prompt>\n");
        fprintf(stderr, "Example: ./seq 4 42 \"What is Fibonacci Number?\"\n");
        fprintf(stderr, "Note:    <prompt> must be quoted with \"\", only one prompt supported\n");
        exit(1);
    }

    // parameter validation/overrides
    if (num_thr <= 0 || num_thr > 16) { 
        fprintf(stderr, "num_thr must between 1 and 16 \n");
        exit(EXIT_FAILURE);
    }
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);

    // build the Transformer via the model .bin file
    build_transformer(&transformer, model_path);
    // build the Tokenizer via the tokenizer .bin file
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    // build the Sampler
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // initialize thread pool
    init_thr_pool(num_thr);

    printf("user: %s \n", prompt);
    // perform multi-threading generation
    generate(prompt);
    
    // close thread pool
    close_thr_pool();

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}