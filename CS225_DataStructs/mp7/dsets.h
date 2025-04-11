/**
 * @file dsets.h
 * Definition of disjoint set class, which is based on up-tree.
 * Path compression and Union by size strategy used.
 *
 * @author Yu Siying
 * @date Summer 2023
 */

#ifndef DSETS_H
#define DSETS_H

#include <vector>
using std::vector;

/**
 * DisjointSets Class. Provides interface for constructing, inserting
 * and finding elements in up-tree.
 * Each DisjointSets object represents a family of disjoint sets,
 * where each element has an integer index. 
 **/
class DisjointSets{
    private:
        /**
         * Uptree used to store the elements. Index is 0-based.
         * Each element of the vector represents a node. A nonnegative number is 
         * the index of the parent of the current node; a negative number in a 
         * root node is the negative of the set size.
         **/
        std::vector<int> uptree;
    public:
        /**
         * Creates n unconnected root nodes at the end of the vector.
         * Parameter: num -- The number of nodes to create.
         **/
        void addelements(int num);
        /**
         * Compress paths and works as described in lecture.
         * Parameter: elem -- the target element.
         * Return: Nonnegative value: The index of the root of the up-tree in 
         *         which the parameter element resides.
         *         -1: The element doesn't exist in current sets.
         **/
        int find(int elem);
        /**
         * Union two disjoint sets by size.
         * Parameter: a	-- Index of the first element to union.
         *            b -- Index of the second element to union.
         **/
        void setunion(int a, int b);
        /**
         * Find out the size of current disjoint set
         * Parameter: elem - One element of current disjoint set
         * Return: The size of current disjoint set
         **/
        int size(int elem);
};

#endif