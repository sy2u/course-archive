/**
 * @file quackfun.cpp
 * This is where you will implement the required functions for the
 *  stacks and queues portion of the lab.
 */

namespace QuackFun {

/**
 * Sums items in a stack.
 * @param s A stack holding values to sum.
 * @return The sum of all the elements in the stack, leaving the original
 *  stack in the same state (unchanged).
 *
 * @note You may modify the stack as long as you restore it to its original
 *  values.
 * @note You may use only two local variables of type T in your function.
 *  Note that this function is templatized on the stack's type, so stacks of
 *  objects overloading the + operator can be summed.
 * @note We are using the Standard Template Library (STL) stack in this
 *  problem. Its pop function works a bit differently from the stack we
 *  built. Try searching for "stl stack" to learn how to use it.
 * @hint Think recursively!
 */
template <typename T>
T sum(stack<T>& s){
    if(s.empty()){
        return 0;
    } else {
        T val_top = s.top();
        s.pop();
        T val_sum = sum(s);
        s.push(val_top);
        return val_sum+val_top;
    }

}

/**
 * Reverses even sized blocks of items in the queue. Blocks start at size
 * one and increase for each subsequent block.
 * @param q A queue of items to be scrambled
 *
 * @note Any "leftover" numbers should be handled as if their block was
 *  complete.
 * @note We are using the Standard Template Library (STL) queue in this
 *  problem. Its pop function works a bit differently from the stack we
 *  built. Try searching for "stl stack" to learn how to use it.
 * @hint You'll want to make a local stack variable.
 */
template <typename T>
void scramble(queue<T>& q)
{
    stack<T> s;
    queue<T> q2;
    int length = q.size();
    int num_block = 0;
    for( int index = 0; index < length; num_block++ ){
        for( int i = 0; i < num_block; i++ ){
            if( !q.empty() ){
                q2.push(q.front());
                q.pop();
            }
            index++;
        }
    }
    for( int i = 1; i <= num_block; i++ ){
        if( i%2 == 0 ){
            for( int j = 0; j < i ; j++ ){
                if( !q2.empty() ){
                    s.push(q2.front());
                    q2.pop();
                }
            }
            for( int j = 0; j < i; j++ ){
                if( !s.empty() ){
                    q.push(s.top());
                    s.pop();
                }
            }
        } else {
            for( int j = 0; j < i; j++ ){
                if( !q2.empty() ){
                    q.push(q2.front());
                    q2.pop();
                }
            }
        }
    }
    
}

/**
 * @return true if the parameter stack and queue contain only elements of
 *  exactly the same values in exactly the same order; false, otherwise.
 *
 * @note You may assume the stack and queue contain the same number of items!
 * @note There are restrictions for writing this function.
 * - Your function may not use any loops
 * - In your function you may only declare ONE local boolean variable to use in
 *   your return statement, and you may only declare TWO local variables of
 *   parametrized type T to use however you wish.
 * - No other local variables can be used.
 * - After execution of verifySame, the stack and queue must be unchanged. Be
 *   sure to comment your code VERY well.
 */
template <typename T>
bool verifySame(stack<T>& s, queue<T>& q)
{
    bool retval = true;
    if( s.empty() ){
        retval = true;
    } else {
        T val_s = s.top(); // hold stack value
        s.pop();
        retval = verifySame(s,q);
        T val_q = q.front(); // hold queue value
        q.pop();
        s.push(val_s); // restore orignal stack
        q.push(val_q); // restore orignal queue
        retval &= (val_s==val_q);
    }
    return retval;
}

}
