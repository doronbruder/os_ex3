#include <pthread.h>
#include <atomic>
#include <algorithm>
#include <semaphore.h>
#include "Barrier.h"
#include "MapReduceFramework.h"
#include <iostream>
#include <bitset>
#include <dmapi_types.h>

typedef std::vector<IntermediatePair> IntermediateVec;
typedef std::vector<pthread_mutex_t *> mutexVec;
typedef std::vector<K2 *> ReduceKeys;
typedef std::vector<V2 *> mapVec;
typedef struct jobContext jobContext;


typedef struct ThreadContext
{
    int threadID;
    jobContext *job;
} ThreadContext;


typedef struct jobContext
{
    int numOfThreads;
    const MapReduceClient *client;
    const InputVec *inputVec;
    std::vector<std::pair<K2 *, V2 *>> *allIntermediateVec;
    OutputVec *outputVec;
    ThreadContext *contexts;
    pthread_t *threads;
    Barrier *barrier;
    std::atomic<unsigned long> *map_sync_counter;
    std::atomic<unsigned long> *reduce_sync_counter;
    std::atomic<unsigned long> *total_shuffle_work;
    std::atomic<unsigned long> *map_amount;
    std::atomic<unsigned long> *shuffle_so_far;
    std::atomic<unsigned long> *reducing_amount;
    std::atomic<unsigned long> *thread_mapped;
    stage_t stage;
    IntermediateMap *intermediateMap;
    pthread_mutex_t *stateMutex;
    pthread_mutex_t *processMutex;
    mutexVec *intermediateMutexes;
    pthread_mutex_t *outputMutex;
    ReduceKeys *reduceKeys;
//    JobState *state;
//    std::atomic<uint64_t> *atomic_process;
} jobContext;

void mutexLockWithErrors(pthread_mutex_t *mutex)
{
    if (pthread_mutex_lock(mutex) != 0)
    {
        exit(1);
    }
}

void mutexUnlockWithErrors(pthread_mutex_t *mutex)
{
    if (pthread_mutex_unlock(mutex) != 0)
    {
        exit(1);
    }
}

void setAtomicProcess(jobContext *job, stage_t stage)
{
    mutexLockWithErrors(job->stateMutex);
    job->stage = stage;
////    std::string binary = std::bitset<64>((*(job->atomic_process))).to_string();
////    std::cerr<<"before"<<(binary)<<std::endl;
//    (*(job->atomic_process)) += stage;
//    (*(job->atomic_process)) = (*(job->atomic_process)) << (unsigned long)(31);
////    binary = std::bitset<64>((*(job->atomic_process))).to_string();
////    std::cerr<<"after stage"<<(binary)<<std::endl;
////    binary = std::bitset<64>((*(job->atomic_process))).to_string();
//    (*(job->atomic_process)) += total;
//    (*(job->atomic_process)) = (*(job->atomic_process)) << (unsigned long)(31);
////    std::cerr<<"after shift"<<binary<<std::endl;
////    (*(job->atomic_process) << 31);
//    *(job->atomic_process) += start;
////    binary = std::bitset<64>((*(job->atomic_process))).to_string();
////    std::cerr<<"after start"<<binary<<std::endl;
    mutexUnlockWithErrors(job->stateMutex);
//    job->state->stage = stage;
//    job->state->percentage = start/total;
}

void frameworkMap(ThreadContext *tc)
{
    jobContext *job = tc->job;
    unsigned long inputVectorLength = job->inputVec->size();
    unsigned long oldValue = (*(job->map_sync_counter))++;
    while (oldValue < inputVectorLength)
    {
        K1 *key = job->inputVec->at(oldValue).first;
        V1 *value = job->inputVec->at(oldValue).second;
        job->client->map(key, value, tc);
//        (*(job->atomic_process))++;
        (*(job->map_amount))++;

        //mutex this calculation (??)
//        job->state->percentage = float(*(job->mapping_amount))/float(inputVectorLength);
        oldValue = (*(job->map_sync_counter))++;
    }
}


void *frameworkShuffle(void *arg)
{
    auto *tc = (ThreadContext *) arg;
    jobContext *job = tc->job;
    //while we didn't mapping all the inputs
    while (int(*(job->thread_mapped)) < tc->job->numOfThreads - 1)
    {
        //iterate over all the therads
        for (int i = 0; i < job->numOfThreads - 1; i++)
        {
            mutexLockWithErrors(tc->job->intermediateMutexes->at(i));
            auto curVec = &job->allIntermediateVec[i];
            //iterate over all the pairs in the current
            //thread intermediate vector and map them
            while (!curVec->empty())
            {
                IntermediatePair pair = curVec->back();
                curVec->pop_back();
                if (job->intermediateMap->find(pair.first) == job->intermediateMap->end())
                {
//                    std::vector<V2 *> vec{pair.second};
                    auto *toAdd = new mapVec;
                    toAdd->push_back(pair.second);
                    job->intermediateMap->insert({pair.first, *(toAdd)});
                }
                else
                {
                    job->intermediateMap->at(pair.first).push_back(pair.second);
                }
                (*(job->shuffle_so_far))++;
            }
            mutexUnlockWithErrors(tc->job->intermediateMutexes->at(i));
        }
    }
    setAtomicProcess(job, SHUFFLE_STAGE);
//    job->state->stage = SHUFFLE_STAGE;
    //mutex this calculation (??)
//    job->state->percentage = (*(job->shuffle_amount))/inputVectorLength;
    for (int i = 0; i < job->numOfThreads - 1; i++)
    {
        auto *curVec = &(job->allIntermediateVec[i]);
        while (!curVec->empty())
        {
            IntermediatePair pair = curVec->back();
            curVec->pop_back();
            if (job->intermediateMap->find(pair.first) == job->intermediateMap->end())
            {
//                std::vector<V2 *> vec{pair.second};
                auto *toAdd = new mapVec;
                toAdd->push_back(pair.second);
                job->intermediateMap->insert({pair.first, *toAdd});
            }
            else
            {
                job->intermediateMap->at(pair.first).push_back(pair.second);
            }
//            (*(job->atomic_process))++;
            (*(job->shuffle_so_far))++;

        }
    }
//    job->state->percentage = (*(job->shuffle_amount))/inputVectorLength;
    for (auto &it : *(job->intermediateMap))
    {
        tc->job->reduceKeys->push_back(it.first);
    }
//    tc->job->barrier->barrier();
    return nullptr;
}


void frameworkReduce(ThreadContext *tc)
{
    jobContext *job = tc->job;
    unsigned long len = job->reduceKeys->size();
    unsigned long oldValue = (*(job->reduce_sync_counter))++;
    while (oldValue < len)
    {
        K2 *key = job->reduceKeys->at(oldValue);
        auto map = *(job->intermediateMap);
        mapVec value = map[key];
        job->client->reduce(key, value, tc);
        (*(job->reducing_amount))++;
        oldValue = (*(job->reduce_sync_counter))++;
    }
}

void *threadWork(void *arg)
{
    auto *threadContext = (ThreadContext *) arg;
    setAtomicProcess(threadContext->job, MAP_STAGE);
    if (threadContext->threadID == threadContext->job->numOfThreads - 1)
    {
        frameworkShuffle(threadContext);
    }
    else
    {
        frameworkMap(threadContext);
        (*(threadContext->job->thread_mapped))++;
    }
    threadContext->job->barrier->barrier();
    //STOP TILL THE SHUFFLE FINISH

    //From here down is after the shuffle stage
    setAtomicProcess(threadContext->job, REDUCE_STAGE);

    frameworkReduce(threadContext);
    return nullptr;
}


JobHandle startMapReduceJob(const MapReduceClient &client,
                            const InputVec &inputVec, OutputVec &outputVec,
                            int multiThreadLevel)
{
    auto *currJobContext = new jobContext;
    auto *barrier = new Barrier(multiThreadLevel);
    auto *map_sync_counter = new std::atomic<unsigned long>(0);
    auto *reduce_sync_counter = new std::atomic<unsigned long>(0);
    auto *atomic_mapping_counter = new std::atomic<unsigned long>(0);
    auto *reducing_amount = new std::atomic<unsigned long>(0);
    currJobContext->inputVec = &inputVec;
    currJobContext->outputVec = &outputVec;
    currJobContext->contexts = new ThreadContext[multiThreadLevel];
    currJobContext->threads = new pthread_t[multiThreadLevel];
    currJobContext->intermediateMutexes = new mutexVec();
    currJobContext->outputMutex = new pthread_mutex_t();
    currJobContext->intermediateMap = new IntermediateMap;
    currJobContext->numOfThreads = multiThreadLevel;
    currJobContext->allIntermediateVec = new std::vector<std::pair<K2 *, V2 *>>[multiThreadLevel];
    currJobContext->reduceKeys = new ReduceKeys;
    currJobContext->client = &client;
    currJobContext->barrier = barrier;
    currJobContext->map_sync_counter = map_sync_counter;
    currJobContext->map_amount = atomic_mapping_counter;
    currJobContext->reduce_sync_counter = reduce_sync_counter;
    currJobContext->reducing_amount = reducing_amount;
    currJobContext->stateMutex = new pthread_mutex_t();
    currJobContext->processMutex = new pthread_mutex_t();
    currJobContext->total_shuffle_work = new std::atomic<unsigned long>(0);
    currJobContext->shuffle_so_far = new std::atomic<unsigned long>(0);
    currJobContext->thread_mapped = new std::atomic<unsigned long>(0);
    currJobContext->stage = UNDEFINED_STAGE;
    for (int i = 0; i < multiThreadLevel; i++)
    {
        currJobContext->contexts[i] = {i, currJobContext};
        currJobContext->intermediateMutexes->push_back(new pthread_mutex_t());
    }
    for (int i = 0; i < multiThreadLevel; ++i)
    {
        pthread_create(currJobContext->threads + i, nullptr, threadWork,
                       currJobContext->contexts + i);
    }
    return currJobContext;
}


void emit2(K2 *key, V2 *value, void *context)
{
    auto *tc = (ThreadContext *) context;
    mutexLockWithErrors(tc->job->intermediateMutexes->at(tc->threadID));
    tc->job->allIntermediateVec[tc->threadID].emplace_back(key, value);
    mutexUnlockWithErrors(tc->job->intermediateMutexes->at(tc->threadID));
    (*(tc->job->total_shuffle_work))++;
}

void emit3(K3 *key, V3 *value, void *context)
{
    auto *tc = (ThreadContext *) context;
    mutexLockWithErrors(tc->job->outputMutex);
    tc->job->outputVec->push_back({key, value});
    mutexUnlockWithErrors(tc->job->outputMutex);
}

void waitForJob(JobHandle job)
{
    auto *jc = (jobContext *) job;
    for (int i = 0; i < jc->numOfThreads; i++)
    {
        if (pthread_join(jc->threads[i], nullptr) != 0)
        { //merges threads
            exit(1);
        }
    }
}

void getJobState(JobHandle job, JobState *state)
{
    auto *curJob = (jobContext *) job;
    mutexLockWithErrors(curJob->stateMutex);
    if (curJob->stage == UNDEFINED_STAGE)
    {
        state->percentage = 0.0;
    }
    else if (curJob->stage == MAP_STAGE)
    {
        state->percentage = (float) (*(curJob->map_amount)) / (float) (curJob->inputVec->size());
    }
    else if (curJob->stage == SHUFFLE_STAGE)
    {
        state->percentage =
                (float) (*(curJob->shuffle_so_far)) / (float) (*(curJob->total_shuffle_work));
    }
    else
    {
        state->percentage =
                (float) (*(curJob->reducing_amount)) / (float) (curJob->intermediateMap->size());
    }
    state->percentage *= 100;
    state->stage = curJob->stage;
    mutexUnlockWithErrors(curJob->stateMutex);
}

void closeJobHandle(JobHandle job)
{
    auto *jc = (jobContext *) job;
    waitForJob(jc);
    for (auto& mut:*(jc->intermediateMutexes))
    {
        pthread_mutex_destroy(mut);
    }
    for(auto& interVec:*(jc->allIntermediateVec)){
        delete interVec.second;
    }
    delete[] jc->allIntermediateVec;
    jc->intermediateMutexes->clear();
    delete jc->intermediateMutexes;
    delete jc->outputMutex;
    delete jc->intermediateMap;
    jc->reduceKeys->clear();
    delete jc->reduceKeys;
    delete jc->stateMutex;
    delete jc->processMutex;
    delete jc->total_shuffle_work;
    delete jc->shuffle_so_far;
    delete jc->thread_mapped;
    delete jc->barrier;
    delete jc->map_sync_counter;
    delete jc->reduce_sync_counter;
    delete jc->map_amount;
    delete jc->reducing_amount;
    delete[] jc->threads;
    delete[] jc->contexts;
    delete jc;
}