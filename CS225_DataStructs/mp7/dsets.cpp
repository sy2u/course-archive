/**
 * @file dsets.cpp
 * Implementation of disjoint set class, which is based on up-tree.
 * Path compression and Union by size strategy used.
 *
 * @author Yu Siying
 * @date Summer 2023
 */

#include "dsets.h"

/**
 * Creates n unconnected root nodes at the end of the vector.
 * @param num -- the number of nodes to create.
 **/
void DisjointSets::addelements(int num){
    for( int i = 0; i < num; i++ ){
        uptree.push_back(-1);
    }
}

/**
 * Compress paths and works as described in lecture.
 * @param elem -- the target element.
 * @return Nonnegative value: The index of the root of the up-tree in 
 *         which the parameter element resides.
 *         -1: The element doesn't exist in current sets.
 **/
int DisjointSets::find(int elem){
    if( elem >= uptree.size() ){ return -1; } // elem doesn't exist
    if( uptree[elem] < 0 ){ return elem; }    // base case: root found
    uptree[elem] = find(uptree[elem]);        // update predecessor
    return uptree[elem];
}

/**
 * Union two disjoint sets by size.
 * @param a	-- Index of the first element to union.
 * @param b -- Index of the second element to union.
 **/
void DisjointSets::setunion(int a, int b){
    int root1, root2; 
    if( size(a)>=size(b) ){
        root1 = find(a); root2 = find(b);
    } else { 
        root1 = find(b); root2 = find(a);
    } // join root2 to root 1
    if( root1 == root2 ){ return; }
    uptree[root1] = uptree[root1]+uptree[root2];
    uptree[root2] = root1;
}

/**
 * Find out the size of current disjoint set
 * @param elem -- One element of current disjoint set
 * @return The size of current disjoint set
 **/
int DisjointSets::size(int elem){
    if( elem >= uptree.size() ){ return -1; }  // elem doesn't exist
    int root = find(elem);
    return -uptree[root];
}