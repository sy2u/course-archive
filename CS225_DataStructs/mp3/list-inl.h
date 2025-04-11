/**
 * @file list.cpp
 * Doubly Linked List (MP 3).
 */

/**
 * Destroys the current List. This function should ensure that
 * memory does not leak on destruction of a list.
 */
template <class T>
List<T>::~List() {
  /// @todo Graded in MP3.1
  clear();
}

/**
 * Destroys all dynamically allocated memory associated with the current
 * List class.
 */
template <class T>
void List<T>::clear() {
  /// @todo Graded in MP3.1
  ListNode* next;
  ListNode* curr = head_;
  while( curr != NULL ){
    next = curr->next;
    delete curr;
    curr = next;
  }
}

/**
 * Inserts a new node at the front of the List.
 * This function **SHOULD** create a new ListNode.
 *
 * @param ndata The data to be inserted.
 */
template <class T>
void List<T>::insertFront(T const& ndata) {
  /// @todo Graded in MP3.1
  ListNode* newnode = new ListNode(ndata);
  if( head_ != NULL ){ head_->prev = newnode; }
  if( tail_ == NULL ){ tail_ = newnode; }
  newnode->next = head_;
  head_ = newnode;
  length_++;
}

/**
 * Inserts a new node at the back of the List.
 * This function **SHOULD** create a new ListNode.
 *
 * @param ndata The data to be inserted.
 */
template <class T>
void List<T>::insertBack(const T& ndata) {
  /// @todo Graded in MP3.1
  ListNode* newnode = new ListNode(ndata);
  if( tail_ != NULL ){ tail_->next = newnode; }
  if( head_ == NULL ){ head_ = newnode; }
  newnode->prev = tail_;
  tail_ = newnode;
  length_++;
}

/**
 * Reverses the current List.
 */
template <class T>
void List<T>::reverse() {
  reverse(head_, tail_);
}

/**
 * Helper function to reverse a sequence of linked memory inside a List,
 * starting at startPoint and ending at endPoint. You are responsible for
 * updating startPoint and endPoint to point to the new starting and ending
 * points of the rearranged sequence of linked memory in question.
 *
 * @param startPoint A pointer reference to the first node in the sequence
 *  to be reversed.
 * @param endPoint A pointer reference to the last node in the sequence to
 *  be reversed.
 */
template <class T>
void List<T>::reverse(ListNode*& startPoint, ListNode*& endPoint) {
    /// @todo Graded in MP3.1
    if( (startPoint==endPoint) || (startPoint==NULL) ){ return; }
    ListNode* traverse = startPoint->next;
    while( traverse != endPoint ){
      ListNode* orig_n = traverse->next;
      traverse->next = traverse->prev;
      traverse->prev = orig_n;
      traverse = orig_n;
    }
    ListNode* orig_s = startPoint;
    ListNode* orig_e_next = endPoint->next;
    startPoint = endPoint;
    startPoint->next = endPoint->prev;
    startPoint->prev = orig_s->prev;
    if( startPoint->prev != NULL ){ startPoint->prev->next = startPoint; }
    endPoint = orig_s;
    endPoint->prev = orig_s->next;
    endPoint->next = orig_e_next;
    if( endPoint->next != NULL ){ endPoint->next->prev = endPoint; }
}

/**
 * Reverses blocks of size n in the current List. You should use your
 * reverse( ListNode * &, ListNode * & ) helper function in this method!
 *
 * @param n The size of the blocks in the List to be reversed.
 */
template <class T>
void List<T>::reverseNth(int n) {
  /// @todo Graded in MP3.1
  if( n >= length_ ){ 
    reverse(); 
  } else {
    int left = length_;
    ListNode* startPoint = head_;
    ListNode* endPoint = head_;
    while( left > 0 ){
      for( int i = 0; i < n-1; i++ ){
        if( endPoint != tail_ ){
          endPoint = endPoint->next;
        }
      }
      if( startPoint == head_ ){ reverse(head_,endPoint); } 
      else if( endPoint == tail_ ){ reverse(startPoint,tail_); }
      else { reverse(startPoint,endPoint); }
      endPoint = endPoint->next;
      startPoint = endPoint;
      left -= n;
    }
  }
}

/**
 * Modifies the List using the waterfall algorithm.
 * Every other node (starting from the second one) is removed from the
 * List, but appended at the back, becoming the new tail. This continues
 * until the next thing to be removed is either the tail (**not necessarily
 * the original tail!**) or NULL.  You may **NOT** allocate new ListNodes.
 * Note that since the tail should be continuously updated, some nodes will
 * be moved more than once.
 */
template <class T>
void List<T>::waterfall() {
  /// @todo Graded in MP3.1
  if( length_ == 0 ){ return; }
  ListNode* next;
  ListNode* curr = head_->next;
  while( (curr!=tail_) && (curr!=NULL) ){
    next = curr->next->next;
    curr->prev->next = curr->next;
    curr->next->prev = curr->prev;
    tail_->next = curr;
    curr->prev = tail_;
    curr->next = NULL;
    tail_ = curr;
    curr = next;
  }
}

/**
 * Splits the given list into two parts by dividing it at the splitPoint.
 *
 * @param splitPoint Point at which the list should be split into two.
 * @return The second list created from the split.
 */
template <class T>
List<T> List<T>::split(int splitPoint) {
    if (splitPoint > length_)
        return List<T>();

    if (splitPoint < 0)
        splitPoint = 0;

    ListNode* secondHead = split(head_, splitPoint);

    int oldLength = length_;
    if (secondHead == head_) {
        // current list is going to be empty
        head_ = NULL;
        tail_ = NULL;
        length_ = 0;
    } else {
        // set up current list
        tail_ = head_;
        while (tail_->next != NULL)
            tail_ = tail_->next;
        length_ = splitPoint;
    }

    // set up the returned list
    List<T> ret;
    ret.head_ = secondHead;
    ret.tail_ = secondHead;
    if (ret.tail_ != NULL) {
        while (ret.tail_->next != NULL)
            ret.tail_ = ret.tail_->next;
    }
    ret.length_ = oldLength - splitPoint;
    return ret;
}

/**
 * Helper function to split a sequence of linked memory at the node
 * splitPoint steps **after** start. In other words, it should disconnect
 * the sequence of linked memory after the given number of nodes, and
 * return a pointer to the starting node of the new sequence of linked
 * memory.
 *
 * This function **SHOULD NOT** create **ANY** new List objects!
 *
 * @param start The node to start from.
 * @param splitPoint The number of steps to walk before splitting.
 * @return The starting node of the sequence that was split off.
 */
template <class T>
typename List<T>::ListNode* List<T>::split(ListNode* start, int splitPoint) {
    /// @todo Graded in MP3.2
    ListNode* traverse = start;
    for( int i = 0; i < splitPoint; i++ ){
      if( traverse != NULL ){
        traverse = traverse->next;
      } else { return NULL; }
    }
    if( traverse == NULL ){ return NULL; }
    if( traverse->prev != NULL ){
      traverse->prev->next = NULL;
      tail_ = traverse->prev;
    }
    return traverse;
}

/**
 * Merges the given sorted list into the current sorted list.
 *
 * @param otherList List to be merged into the current list.
 */
template <class T>
void List<T>::mergeWith(List<T>& otherList) {
    // set up the current list
    head_ = merge(head_, otherList.head_);
    tail_ = head_;

    // make sure there is a node in the new list
    if (tail_ != NULL) {
        while (tail_->next != NULL)
            tail_ = tail_->next;
    }
    length_ = length_ + otherList.length_;

    // empty out the parameter list
    otherList.head_ = NULL;
    otherList.tail_ = NULL;
    otherList.length_ = 0;
}

/**
 * Helper function to merge two **sorted** and **independent** sequences of
 * linked memory. The result should be a single sequence that is itself
 * sorted.
 *
 * This function **SHOULD NOT** create **ANY** new List objects.
 *
 * @param first The starting node of the first sequence.
 * @param second The starting node of the second sequence.
 * @return The starting node of the resulting, sorted sequence.
 */
template <class T>
typename List<T>::ListNode* List<T>::merge(ListNode* first, ListNode* second) {
  /// @todo Graded in MP3.2
  ListNode* retval;
  ListNode* base;
  ListNode* insert;
  ListNode* insertnext;
  ( first->data < second->data ) ? ( base = first ) : ( base = second );
  ( first->data < second->data ) ? ( insert = second ) : ( insert = first );
  retval = base;
  while( (base->next!=NULL) && (insert!=NULL) ){
    if( insert->data < base->next->data ){
      insertnext = insert->next;
      insert->next = base->next;
      if( base->next != NULL ){ base->next->prev = insert; }
      insert->prev = base;
      base->next = insert;
      insert = insertnext;
    } else {
      base = base->next;
    }
  }
  if( insert != NULL ){
    base->next = insert;
    insert->prev = base;
  }
  return retval;
}

/**
 * Sorts the current list by applying the Mergesort algorithm.
 */
template <class T>
void List<T>::sort() {
    if (empty())
        return;
    head_ = mergesort(head_, length_);
    tail_ = head_;
    while (tail_->next != NULL)
        tail_ = tail_->next;
}

/**
 * Sorts a chain of linked memory given a start node and a size.
 * This is the recursive helper for the Mergesort algorithm (i.e., this is
 * the divide-and-conquer step).
 *
 * @param start Starting point of the chain.
 * @param chainLength Size of the chain to be sorted.
 * @return A pointer to the beginning of the now sorted chain.
 */
template <class T>
typename List<T>::ListNode* List<T>::mergesort(ListNode* start, int chainLength) {
    /// @todo Graded in MP3.2
    if( chainLength == 1 ){
      start->prev = NULL;
      start->next = NULL;
      return start;
    } else {
      ListNode* halfstart = start;
      for( int i = 0; i < chainLength/2; i++ ){
        halfstart = halfstart->next;
      }
      ListNode* first = mergesort(start,chainLength/2);
      ListNode* second = mergesort(halfstart,chainLength-chainLength/2);
      return merge(first,second);
    }
}
