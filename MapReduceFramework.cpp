#include "Barrier.h"
#include "MapReduceClient.h"
#include <algorithm>
#include <atomic>
#include <iostream>
#include <queue>
#include <semaphore.h>

//ERROR DEFINITIONS
#define PT_MUTEX_DESTROY_ERROR "[[MapReduceFramework]] error on pthread_mutex_destroy"
#define PT_MUTEX_LOCK_ERROR "[[MapReduceFramework]] error on pthread_mutex_lock"
#define PT_MUTEX_UNLOCK_ERROR "[[MapReduceFramework]] error on pthread_mutex_unlock"
#define PT_JOIN_ERROR "[[MapReduceFramework]] error on pthread_join"
#define PT_CREATE_ERROR "[[MapReduceFramework]] error on pthread_create"
#define SEM_POST "[[MapReduceFramework]] error on sem_post"
#define SEM_DESTROY_ERROR "[[MapReduceFramework]] error on sem_destroy"
#define SEM_WAIT_ERROR "[[MapReduceFramework]] error on sem_wait"


/*
 * Structure which holds the data each thread needs for operating the framework.
 */
struct threadContext
{
    std::atomic<int> *atomicCounter; // pointer to atomic counter used in map phase
    Barrier *barrier; // pointer to barrier used to sync the sort phase
    const MapReduceClient *client; // holds the task that the framework should run
    int tid; // the thread ID
    const InputVec *inputVec; // pointer to the input vector
    IntermediateVec *intermediateVec; // pointer to an intermediate vector;
    std::vector<IntermediateVec> *sortedVectors; // pointer to vector intermediate sorted vectors;
    std::vector<IntermediateVec> *shuffledVectors; // pointer to vector intermediate shuffled
    // vectors;
    OutputVec *outputVec; // pointer to the output vector
    sem_t *reduceSemaphore; // pointer to the semaphore used in reduce phase
    pthread_mutex_t *reduceShuffleMutex; // pointer to a mutex used in reduce and shuffle phases
    pthread_mutex_t *outputMutex; // pointer to a mutex used in emit3 phase
};

// Function Declarations

bool comparator(IntermediatePair firstPair, IntermediatePair secondPair);

IntermediateVec createShuffledVector(threadContext *context, K2 *maxKey);

void destroyMutexes(threadContext *context);

void destroySemaphore(threadContext *context);

void eraseEmptySortedVectors(threadContext *context);

K2 *findMaxKey(threadContext *context);

void map(threadContext *context);

void reduce(threadContext *context);

void shuffle(threadContext *);

void sort(threadContext *context);

void *threadRoutine(void *arg);

/**
 * In case of failure this function is called to destroy the mutexes of the framework
 * @param context the data which the current running thread holds.
 */
void destroyMutexes(threadContext *context)
{
    if (pthread_mutex_destroy(context->reduceShuffleMutex))
    {
        std::cerr << PT_MUTEX_DESTROY_ERROR;
        exit(EXIT_FAILURE);
    }
    if (pthread_mutex_destroy(context->outputMutex))
    {
        std::cerr << PT_MUTEX_DESTROY_ERROR;
        exit(EXIT_FAILURE);
    }
}

/**
 * In case of failure this function is called to destroy the semaphore of the framework
 * @param context the data which the current running thread holds.
 */
void destroySemaphore(threadContext *context)
{
    if (sem_destroy(context->reduceSemaphore))
    {
        std::cerr << SEM_DESTROY_ERROR;
        exit(EXIT_FAILURE);
    }
}

/**
 * In this phase each thread reads pairs of (k1,v1) from the input vector and calls the map
 * function given by the client on each of them. The map function in turn will call the emit2
 * function to output (k2,v2) pairs.
 * @param context the data which the current running thread holds.
 */
void map(threadContext *context)
{
    int index;
    const InputVec inVec = *context->inputVec;
    while ((unsigned int) *context->atomicCounter < inVec.size())
    {
        index = (*(context->atomicCounter))++;
        auto *key = inVec[index].first;
        auto *value = inVec[index].second;
        context->client->map(key, value, context->intermediateVec);
    }
}

/**
 * Help function to the sort function, describes the rule for sorting the intermediate vecor
 * @param firstPair an intermediate pair (K2,V2)
 * @param secondPair an intermediate pair (K2,V2)
 * @return whether the key in the first pair given is greater than the key in the second pair
 */
bool comparator(IntermediatePair firstPair, IntermediatePair secondPair)
{
    return (*firstPair.first < *secondPair.first);
}

/**
 * Sorts the threads intermediate vector according to the keys within.
 * @param context the data which the current running thread holds.
 */
void sort(threadContext *context)
{
    std::sort(context->intermediateVec->begin(), context->intermediateVec->end(), comparator);
}

/**
 * finds the value of the max key in the sorted vectors under the assumption the vectors are
 * sorted such that the max key in each vector is in the end(back) of the vector.
 * @param context the data which the current running thread holds.
 * @return pointer to a key that hold the current max value.
 */
K2 *findMaxKey(threadContext *context)
{
    K2 *maxKey = nullptr;
    for (auto &sortedVector : *context->sortedVectors)
    {
        if (!sortedVector.empty())
        {
            maxKey = sortedVector.back().first;
            break;
        }
    }
    for (auto &sortedVector : *context->sortedVectors)
    {
        if (sortedVector.empty())
        {
            continue;
        }
        if (*maxKey < *sortedVector.back().first)
        {
            maxKey = sortedVector.back().first;
        }
    }
    return maxKey;
}

/**
 * Creates a new vector of (k2,v2) where all keys are identical(by the value pointed by K2)
 * @param context the data which the current running thread holds.
 * @param maxKey pointer to the the current max key.
 * @return vector of (k2,v2) where all keys are identical
 */
IntermediateVec createShuffledVector(threadContext *context, K2 *maxKey)
{
    IntermediateVec shuffledVec;
    for (auto &sortedVector : *context->sortedVectors)
    {
        if (!sortedVector.empty())
        {
            while (!(*maxKey < *sortedVector.back().first) &&
                   !(*sortedVector.back().first < *maxKey))
            {
                //std::move converts the pair value to rvalue so that it can be pushed back
                shuffledVec.push_back(std::move(sortedVector.back()));
                sortedVector.pop_back();
                if (sortedVector.empty())
                {
                    break;
                }
            }
        }
    }
    return shuffledVec;
}

/**
 * Erase  empty vectors from the sortedVectors data structure.
 * @param context the data which the current running thread holds.
 */
void eraseEmptySortedVectors(threadContext *context)
{
    unsigned int deleteIndex, offset = 0;
    for (unsigned int i = 0; i < context->sortedVectors->size(); ++i)
    {
        deleteIndex = i - offset;
        if (context->sortedVectors->at(deleteIndex).empty())
        {
            ++offset;
            context->sortedVectors->erase(context->sortedVectors->begin() + deleteIndex);
        }
    }
}

/**
 * Creates new sequences of (k2,v2) where in each sequence all keys are identical and all
 * elements with a given key are in a single sequence.
 * @param context the data which the current running thread holds(in case of shuffle it is the
 * main thread).
 */
void shuffle(threadContext *context)
{
    while (!context->sortedVectors->empty())
    {
        K2 *maxKey = findMaxKey(context);
        IntermediateVec shuffledVec = createShuffledVector(context, maxKey);
        if (!shuffledVec.empty())
        {
            if (pthread_mutex_lock(context->reduceShuffleMutex))
            {
                std::cerr << PT_MUTEX_LOCK_ERROR;
                exit(EXIT_FAILURE);
            }
            context->shuffledVectors->push_back(shuffledVec);
            if (pthread_mutex_unlock(context->reduceShuffleMutex))
            {
                std::cerr << PT_MUTEX_UNLOCK_ERROR;
                exit(EXIT_FAILURE);
            }
            if (sem_post(context->reduceSemaphore))
            {
                std::cerr << SEM_POST;
                exit(EXIT_FAILURE);
            }
        }
        eraseEmptySortedVectors(context);
    }
}

/**
 * The reducing threads will wait for the shuffled vectors to be created by the shuffling thread.
 * once one exists it is poped and used be the client's reduce function which call emit3 to produce
 * (k3,v3) pairs.
 * @param context the data which the current running thread holds.
 */
void reduce(threadContext *context)
{
    while (!context->shuffledVectors->empty())
    {
        if (sem_wait(context->reduceSemaphore))
        {
            std::cerr << SEM_WAIT_ERROR;
            destroyMutexes(context);
            destroySemaphore(context);
            exit(EXIT_FAILURE);
        }
        if (pthread_mutex_lock(context->reduceShuffleMutex))
        {
            std::cerr << PT_MUTEX_LOCK_ERROR;
            destroyMutexes(context);
            destroySemaphore(context);
            exit(EXIT_FAILURE);
        }
        if (context->shuffledVectors->empty())
        {
            if (pthread_mutex_unlock(context->reduceShuffleMutex))
            {
                std::cerr << PT_MUTEX_UNLOCK_ERROR;
                destroyMutexes(context);
                destroySemaphore(context);
                exit(EXIT_FAILURE);
            }
        } else
        {
            //std::move converts the pair value to rvalue so that it can be pushed back
            IntermediateVec shuffledVec = std::move(context->shuffledVectors->back());
            context->shuffledVectors->pop_back();
            pthread_mutex_unlock(context->reduceShuffleMutex);
            context->client->reduce(&shuffledVec, context);
        }
    }
    if (sem_post(context->reduceSemaphore))
    {
        std::cerr << SEM_POST;
        destroyMutexes(context);
        destroySemaphore(context);
        exit(EXIT_FAILURE);
    }
}

/**
 * The routine which each thread executes.
 * @param arg the context arguments which each thread has to hold
 * @return nothing
 */
void *threadRoutine(void *arg)
{
    auto *context = (threadContext *) arg;
    map(context);
    sort(context);
    context->barrier->barrier();
    // Only the main thread enters the shuffle phase
    if (context->tid == 0)
    {
        shuffle(context);
    }
    reduce(context);
    return nullptr;
}


//API Functions

/**
 * Produces a (K2*,V2*) pair.
 * @param key K2 pointer.
 * @param value V2 pointer.
 * @param context the data which the current running thread holds.
 */
void emit2(K2 *key, V2 *value, void *context)
{
    IntermediateVec *intermediateVec;
    // used to get pointers into the framework’s variables and data structures.
    intermediateVec = static_cast<IntermediateVec *>(context);
    IntermediatePair pair = {key, value};
    intermediateVec->push_back(pair);
}

/**
 * Emits to the output vector given by the client output pairs.
 * @param key K3 pointer.
 * @param value V3 pointer.
 * @param context the data which the current running thread holds.
 */
void emit3(K3 *key, V3 *value, void *context)
{
    auto *context3 = static_cast<threadContext *>(context);
    if (pthread_mutex_lock(context3->outputMutex))
    {
        std::cerr << PT_MUTEX_LOCK_ERROR;
        destroyMutexes(context3);
        destroySemaphore(context3);
        exit(EXIT_FAILURE);
    }
    OutputPair pair = {key, value};
    context3->outputVec->push_back(pair);
    if (pthread_mutex_unlock(context3->outputMutex))
    {
        std::cerr << PT_MUTEX_UNLOCK_ERROR;
        destroyMutexes(context3);
        destroySemaphore(context3);
        exit(EXIT_FAILURE);
    }
}

/**
 * This function starts runs the entire MapReduce algorithm.
 * @param client the task that the framework should run.
 * @param inputVec a vector of type std::vector<std::pair<K1*, V1*>>, the input elements.
 * @param outputVec a vector of type std::vector<std::pair<K3*, V3*>>, to which the output
                    elements will be added before returning.
 * @param multiThreadLevel the number of worker threads to be used for running the algorithm.
 */
void
runMapReduceFramework(const MapReduceClient &client, const InputVec &inputVec, OutputVec &outputVec,
                      int multiThreadLevel)
{
    std::atomic<int> atomicCounter(0);
    std::vector<IntermediateVec> shuffledVectors;
    std::vector<IntermediateVec> sortedVectors;

    Barrier barrier(multiThreadLevel);
    threadContext contexts[multiThreadLevel];

    sem_t reduceSemaphore = {};

    pthread_mutex_t outputMutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t reduceShuffleMutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_t threads[multiThreadLevel];

    for (int i = 0; i < multiThreadLevel; ++i)
    {
        IntermediateVec intermediateVec;
        sortedVectors.push_back(intermediateVec);
    }

    for (int tid = 0; tid < multiThreadLevel; ++tid)
    {
        contexts[tid] = {&atomicCounter, &barrier, &client, tid, &inputVec, &sortedVectors[tid],
                         &sortedVectors, &shuffledVectors, &outputVec, &reduceSemaphore,
                         &reduceShuffleMutex, &outputMutex};
    }

    for (int tid = 0; tid < multiThreadLevel; ++tid)
    {
        if (pthread_create(threads + tid, nullptr, threadRoutine, contexts + tid))
        {
            std::cerr << PT_CREATE_ERROR;
            exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < multiThreadLevel; ++i)
    {
        if (pthread_join(threads[i], nullptr))
        {
            std::cerr << PT_JOIN_ERROR;
            exit(EXIT_FAILURE);
        }
    }
}
