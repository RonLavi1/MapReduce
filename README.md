��# MapReduce

Client Overview

We have three types of elements, each having its own key type and value type:
1) Input elements – we denote their key type k1 and value type v1.
2) Intermediary elements – we denote their key type k2 and value type v2.
3) Output elements – we denote their key type k3 and value type v3.
The map function receives a key of type k1 and a value of type v1 as input and produces pairs of (k2,v2).
The reduce function receives a sequence of pairs (k2,v2) as input, where all keys are identical, and produces pairs of (k3,v3).

1) Key/Value classes inheriting from k1,k2,k3 and v1,v2,v3 including a < operator for the
keys, to enable sorting.
2) The map function in the MapReduceClient class with the signature:
void map(const K1* key, const V1* value, void* context) const
This function will produce intermediate pairs by calling the framework function
emit2(K2,V2,context).
The context argument is provided to allow emit2 to receive information from the function
that called map.
3) The reduce function in the MapReduceClient class with the signature:
void reduce(const IntermediateVec* value, void* context) const
IntermediateVec is of type std::vector<std::pair<K2*,V2*>>
All pairs in the vector are expected to have the same key (but not necessarily the same
instances of K2).
This function will produce output pairs by calling the framework function
emit3(K3,V3,context).
The context argument is provided to allow emit3 to receive information from the function
that called reduce.


Framework Interface Overview


The framework interface consists of three functions:
1) runMapReduceFramework – This function starts runs the entire MapReduce algorithm.
void runMapReduceFramework(const MapReduceClient& client,
const InputVec& inputVec, OutputVec& outputVec,
unsigned int multiThreadLevel)
client – The implementation of MapReduceClient or in other words the task that the
framework should run.
inputVec – a vector of type std::vector<std::pair<K1*, V1*>>, the input elements
outputVec – a vector of type std::vector<std::pair<K3*, V3*>>, to which the output
elements will be added before returning. You can assume that outputVec is empty.
multiThreadLevel – the number of worker threads to be used for running the algorithm.

2) emit2 – This function produces a (K2*,V2*) pair. It has the following signature:
void emit2 (K2* key, V2* value, void* context)
The context can be used to get pointers into the framework’s variables and data structures.
Its exact type is implementation dependent.

3) emit3 – This function produces a (K3*,V3*) pair. It has the following signature:
void emit2 (K3* key, V3* value, void* context)
The context can be used to get pointers into the framework’s variables and data structures.
Its exact type is implementation dependent.

In this design all threads except thread 0 run three phases: Map, Sort and Reduce, while thread 0
also runs a special shuffle phase between its Sort and Reduce phases.
Map Phase
In this phase each thread reads pairs of (k1,v1) from the input vector and calls the map function on
each of them. The map function in turn will call the emit2 function to output (k2,v2) pairs. We
have two synchronisation challenges here:
1) Splitting the input values between the threads – this will be done using an atomic variable
shared between the threads, an example of using an atomic variable in this manner is
provided together with the exercise. Read it and run it before continuing.
The variable will be initialised to 0, then each thread will increment the variable and check
its old value. The thread can safely call map on the pair in the index old_value knowing
that no other thread will do so. This is repeated until old_value is after the end of the input
vector, as that means that all pairs have been processed and the Map phase has ended.
2) Prevent output race conditions – This will be done by separating the outputs. We will
create a vector for each thread and then emit2 will just append the new k2,v2 pair into the
calling thread’s vector. Accessing the calling thread’s vector can be done by using the
context argument.

In the end of this phase we have multiThreadLevel vectors of (k2,v2) pairs and all elements in the
input vector were processed.

Sort Phase
Immediately after the Map phase each thread will sort its intermediate vector according to the keys
within.

In the end of this phase we must use a barrier – a synchronisation mechanism that makes sure no
thread continues before all threads arrived at the barrier. Once all threads arrive, the waiting
threads can continue.

After the barrier, one of the threads will move on to the Shuffle phase while the rest will skip it
and move directly to the Reduce phase.

Shuffle Phase
our goal in this phase is to create new sequences of (k2,v2) where in each sequence all
keys are identical and all elements with a given key are in a single sequence.
Since our intermediary vectors are sorted, we know that all elements with the largest key must be
at the back of each vector. Thus, creating the new sequence is simply a matter of popping these
elements from the back of each vector and inserting them to a new vector. Now all elements with
the second largest key are at the back of the vectors so we can repeat the process until the
intermediary vectors are empty.

That is a task that is quite difficult to split efficiently into parallel threads so we run it in parallel
with the Reduce phase instead. Whenever we finish creating a new vector, we put it in a queue for
one of the reducing threads to pop and call reduce on.
Use a vector for the queue (note that it is a vector of vectors), with a semaphore for counting the
number of vectors in it. Whenever a new vector is inserted to the queue we will call sem_post() on
the semaphore to notify the reducing threads that they have pending work. Note that you will also
need a mutex for protecting the access to this queue.
Once all intermediary vectors are empty, the shuffling thread will move on to the Reduce phase.


Reduce Phase
The reducing threads will wait for the shuffled vectors to be created by the shuffling thread. They
can do so by calling sem_wait() on the aforementioned semaphore. Once they wake up, there must
be at least one element in the queue, so now they can pop a vector from the back of the queue and
run reduce on it.
The reduce function in turn will call emit3 to produce (k3,v3) pairs. These are inserted directly
to the output vector (outputVec argument of runMapReduceFramework) under the protection of a
mutex. The emit3 function can access the output vector through its context argument.
Once both the Shuffle phase is finished and the queue is empty, runMapReduceFramework may
return as the task is done.
