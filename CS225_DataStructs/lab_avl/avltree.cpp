/**
 * @file avltree.cpp
 * Definitions of the binary tree functions you'll be writing for this lab.
 * You'll need to modify this file.
 */

template <class K, class V>
V AVLTree<K, V>::find(const K& key) const
{
    return find(root, key);
}

template <class K, class V>
V AVLTree<K, V>::find(Node* subtree, const K& key) const
{
    if (subtree == NULL)
        return V();
    else if (key == subtree->key)
        return subtree->value;
    else {
        if (key < subtree->key)
            return find(subtree->left, key);
        else
            return find(subtree->right, key);
    }
}

template <class K, class V>
void AVLTree<K, V>::rotateLeft(Node*& t)
{
    functionCalls.push_back("rotateLeft"); // Stores the rotation name (don't remove this)
    // your code here
    Node* y = t->right;
    t->right = y->left;
    y->left = t;
    t = y;
    if( t != NULL ){ updateHeight(t); }
}

template <class K, class V>
void AVLTree<K, V>::rotateLeftRight(Node*& t)
{
    functionCalls.push_back("rotateLeftRight"); // Stores the rotation name (don't remove this)
    // Implemented for you:
    rotateLeft(t->left);
    rotateRight(t);
}

template <class K, class V>
void AVLTree<K, V>::rotateRight(Node*& t)
{
    functionCalls.push_back("rotateRight"); // Stores the rotation name (don't remove this)
    // your code here
    Node* y = t->left;
    t->left = y->right;
    y->right = t;
    t = y;
    int initHeight = -1;
    if( t != NULL ){ updateHeight(t); }
}

template <class K, class V>
void AVLTree<K, V>::rotateRightLeft(Node*& t)
{
    functionCalls.push_back("rotateRightLeft"); // Stores the rotation name (don't remove this)
    // your code here
    rotateRight(t->right);
    rotateLeft(t);
}

template <class K, class V>
void AVLTree<K, V>::rebalance(Node*& subtree)
{
    // your code here
    if( subtree == NULL ){ return; }
    // if( subtree->left == NULL && subtree->right == NULL ){ return; }
    // rebalance(subtree->left);
    // rebalance(subtree->right);
    int balence = heightOrNeg1(subtree->right)-heightOrNeg1(subtree->left);
    if( balence == -2 ){
        int leftBalance = heightOrNeg1(subtree->left->right)-heightOrNeg1(subtree->left->left);
        if( leftBalance == -1 ){
            rotateRight(subtree);
        } else { rotateLeftRight(subtree); }
    } else if ( balence == 2 ){
        int rightBalance = heightOrNeg1(subtree->right->right)-heightOrNeg1(subtree->right->left);
        if( rightBalance == 1 ){
            rotateLeft(subtree);
        } else { rotateRightLeft(subtree); }
    }
    subtree->height = 1 + max(heightOrNeg1(subtree->left),heightOrNeg1(subtree->right));
}

template <class K, class V>
void AVLTree<K, V>::insert(const K & key, const V & value)
{
    insert(root, key, value);
}

template <class K, class V>
void AVLTree<K, V>::insert(Node*& subtree, const K& key, const V& value)
{
    // your code here
    if( subtree == NULL ){
        subtree = new Node(key,value);
    } else if ( subtree->key < key ){
        insert(subtree->right,key,value);
        rebalance(subtree);
    } else if ( subtree->key > key ){
        insert(subtree->left,key,value);
        rebalance(subtree);
    }
}

template <class K, class V>
void AVLTree<K, V>::remove(const K& key)
{
    remove(root, key);
}

template <class K, class V>
void AVLTree<K, V>::remove(Node*& subtree, const K& key)
{
    if (subtree == NULL)
        return;
    if (key < subtree->key) {
        // your code here
        remove(subtree->left,key);
    } else if (key > subtree->key) {
        // your code here
        remove(subtree->right,key);
    } else {
        if (subtree->left == NULL && subtree->right == NULL) {
            /* no-child remove */
            // your code here
            delete subtree;
            subtree = NULL;
        } else if (subtree->left != NULL && subtree->right != NULL) {
            /* two-child remove */
            // your code here
            Node** IOP = NULL;
            findIOP(root,key,IOP);
            swap(*IOP,subtree);
            delete *IOP;
            *IOP = NULL;
        } else {
            /* one-child remove */
            // your code here
            Node* trash = subtree;
            if( subtree->left != NULL ){ subtree = subtree->left; }
            else{ subtree = subtree->right; }
            delete trash;
        }
        // your code here
        if( subtree != NULL ){ updateHeight(subtree); }
    }
    rebalance(subtree);
}

template <class K, class V>
void AVLTree<K, V>::findIOP(Node*& subtree, const K key, Node**& IOP){
    if( subtree == NULL ){ return; }
    else{
        findIOP(subtree->left,key,IOP);
        if( subtree->key < key ){
            IOP = &subtree;
        } else if( subtree->key == key ){
            return;
        }
        findIOP(subtree->right,key,IOP);
    }
}

template <class K, class V>
int AVLTree<K,V>::updateHeight(Node* subtree){
    if( subtree == NULL ){
        return -1;
    }
    int height = 1+max(updateHeight(subtree->left),updateHeight(subtree->right));
    subtree->height = height;
    return height;
}