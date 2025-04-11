/**
 * @file kdtree.cpp
 * Implementation of KDTree class.
 */

#include <utility>
#include <algorithm>

using namespace std;

template <int Dim>
bool KDTree<Dim>::smallerDimVal(const Point<Dim>& first,
                                const Point<Dim>& second, int curDim) const
{
    /**
     * @todo Implement this function!
     */
    if( first[curDim] == second[curDim] ){
      return first < second;
    }
    return first[curDim]<second[curDim];

}

template <int Dim>
bool KDTree<Dim>::shouldReplace(const Point<Dim>& target,
                                const Point<Dim>& currentBest,
                                const Point<Dim>& potential) const
{
    /**
     * @todo Implement this function!
     */

    double dist_pot = 0;
    double dist_cur = 0;
    for( int i = 0; i < Dim; i++ ){
      dist_pot += (target[i]-potential[i])*(target[i]-potential[i]);
      dist_cur += (target[i]-currentBest[i])*(target[i]-currentBest[i]);
    }
    if( dist_pot == dist_cur ){
      return potential<currentBest;
    }
    return dist_pot<dist_cur;
}

template <int Dim>
KDTree<Dim>::KDTree(const vector<Point<Dim>>& newPoints)
{
    /**
     * @todo Implement this function!
     */
    if( newPoints.empty() ){
      root = NULL;
      return;
    }
    vector<Point<Dim>> KDvector = newPoints;
    KDvector.assign(newPoints.begin(),newPoints.end());
    root = buildHelper(KDvector,0,0,newPoints.size()-1);
}

template <int Dim>
typename KDTree<Dim>::KDTreeNode* KDTree<Dim>::buildHelper(vector<Point<Dim>>& vector, int depth, int left, int right){
  if( right < 0 || left > right ){ return NULL; } 
  if( left == right ){ 
    KDTreeNode* subroot = new KDTreeNode(vector[left]);
    return subroot;
  }
  int curDim = depth%Dim;
  int median_k = (right-left)/2;
  int median_idx = left + median_k;
  quickSelect(left,right,curDim,median_k,vector);
  KDTreeNode* subroot = new KDTreeNode(vector[median_idx]);
  subroot->left = buildHelper(vector,depth+1,left,median_idx-1);
  subroot->right = buildHelper(vector,depth+1,median_idx+1,right);
  return subroot;
}

template <int Dim>
void KDTree<Dim>::swap(int idx_1, int idx_2, vector<Point<Dim>>& vector){
      Point<Dim> tmp = vector[idx_1];
      vector[idx_1] = vector[idx_2];
      vector[idx_2] = tmp;
}

template <int Dim>
void KDTree<Dim>::quickSelect(int left, int right, int dim, int k, vector<Point<Dim>>& vector){
  if( right < 0 || left > right || left == right ){ return; } 
  if( smallerDimVal(vector[left],vector[right],dim) ){ 
    swap(left,right,vector); 
  }
  double pivot_value = vector[right][dim];
  int pivot_idx = 0; // pivot_idx should point to a value larger than pivot
  for( int i = left+1; i < right; i++ ){
    if( vector[i][dim] < pivot_value ){
      swap(i,left+pivot_idx,vector);
      pivot_idx++;
    }
  }
  swap(right,left+pivot_idx,vector);
  if( pivot_idx == k ){ return; }
  if (pivot_idx < k){
    quickSelect(left+pivot_idx+1,right,dim,k-(pivot_idx)-1,vector);
  } else {
    quickSelect(left,left+pivot_idx-1,dim,k,vector);
  }
}

template <int Dim>
KDTree<Dim>::KDTree(const KDTree& other) {
  /**
   * @todo Implement this function!
   */
  size = other.size;
  root = NULL;
  root = copy(root,other.root);
}

template <int Dim>
const KDTree<Dim>& KDTree<Dim>::operator=(const KDTree& rhs) {
  /**
   * @todo Implement this function!
   */
  clear(root);
  size = rhs.size;
  root = copy(root,rhs.root);
  return *this;
}

template <int Dim>
typename KDTree<Dim>::KDTreeNode* KDTree<Dim>::copy(KDTreeNode* lhs, const KDTreeNode* rhs){
  if( rhs == NULL ){ return NULL; }
  lhs = new KDTreeNode();
  lhs->point = rhs->point;
  lhs->left = copy(lhs->left, rhs->left);
  lhs->right = copy(lhs->right,rhs->right);
  return lhs;
}

template <int Dim>
KDTree<Dim>::~KDTree() {
  /**
   * @todo Implement this function!
   */
  size = 0;
  clear(root);
}

template <int Dim>
void KDTree<Dim>::clear(KDTreeNode* subroot){
  if( subroot==NULL ){ return; }
  clear(subroot->left);
  clear(subroot->right);
  delete subroot;
  subroot = NULL;
}

template <int Dim>
Point<Dim> KDTree<Dim>::findNearestNeighbor(const Point<Dim>& query) const
{
    /**
     * @todo Implement this function!
     */
    Point<Dim> nearest = Point<Dim>();
    double distance = 0;
    bool flag = true;
    findNearestHelper(root,query,nearest,distance,flag,0);
    return nearest;
}

template <int Dim>
void KDTree<Dim>::findNearestHelper(KDTreeNode* potential, const Point<Dim>& query,
                                    Point<Dim>& currpoint, double& distbest, bool& flag, int depth) const{
  if( potential == NULL ){ return; }
  int curDim = depth % Dim;
  if( smallerDimVal(query,potential->point,curDim) ){ // left subtree
    findNearestHelper(potential->left,query,currpoint,distbest,flag,depth+1);
    if( flag==true ){
      distbest = 0;
      for( int i = 0; i < Dim; i++ ){
        distbest += (potential->point[i]-query[i])*(potential->point[i]-query[i]);
      }
      currpoint = potential->point;
      flag = false;
    } else {
      if( shouldReplace(query,currpoint,potential->point) ){
        distbest = 0;
        for( int i = 0; i < Dim; i++ ){
          distbest += (potential->point[i]-query[i])*(potential->point[i]-query[i]);
        }
        currpoint = potential->point;
      }
    }
    if( (potential->point[curDim]-query[curDim])*(potential->point[curDim]-query[curDim]) > distbest ){ return; }
    findNearestHelper(potential->right,query,currpoint,distbest,flag,depth+1);
  } else { // right subtree
    findNearestHelper(potential->right,query,currpoint,distbest,flag,depth+1);
    if( flag==true ){
      distbest = 0;
      for( int i = 0; i < Dim; i++ ){
        distbest += (potential->point[i]-query[i])*(potential->point[i]-query[i]);
      }
      currpoint = potential->point;
      flag = false;
    } else {
      if( shouldReplace(query,currpoint,potential->point) ){
        distbest = 0;
        for( int i = 0; i < Dim; i++ ){
          distbest += (potential->point[i]-query[i])*(potential->point[i]-query[i]);
        }
        currpoint = potential->point;
      }
    }
    if( (potential->point[curDim]-query[curDim])*(potential->point[curDim]-query[curDim]) > distbest ){ return; }
    findNearestHelper(potential->left,query,currpoint,distbest,flag,depth+1);
  }
}