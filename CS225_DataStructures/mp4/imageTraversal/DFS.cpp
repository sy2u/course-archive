#include <iterator>
#include <cmath>
#include <list>
#include <stack>

#include "../cs225/PNG.h"
#include "../Point.h"

#include "ImageTraversal.h"
#include "DFS.h"

/**
 * Initializes a depth-first ImageTraversal on a given `png` image,
 * starting at `start`, and with a given `tolerance`.
 */
DFS::DFS(const PNG & png, const Point & start, double tolerance) {  
  /** @todo [Part 1] */
  png_ = png;
  start_ = start;
  tolerance_ = tolerance;
  isVisited_ = std::vector<bool>(png_.height()*png_.width(),false);
  generatePath(start_);
}

/**
 * Returns an iterator for the traversal starting at the first point.
 */
ImageTraversal::Iterator DFS::begin() {
  /** @todo [Part 1] */
  return ImageTraversal::Iterator(false,start_,this);
}

/**
 * Returns an iterator for the traversal one past the end of the traversal.
 */
ImageTraversal::Iterator DFS::end() {
  /** @todo [Part 1] */
  return ImageTraversal::Iterator(true,start_,this);
}

/**
 * Adds a Point for the traversal to visit at some point in the future.
 */
void DFS::add(const Point & point) {
  /** @todo [Part 1] */
  stack_.push(point);
}

/**
 * Removes and returns the current Point in the traversal.
 */
Point DFS::pop() {
  /** @todo [Part 1] */
  Point curr = path_.front();
  path_.pop();
  return curr;
}

/**
 * Returns the current Point in the traversal.
 */
Point DFS::peek() const {
  /** @todo [Part 1] */
  return path_.front();
}

/**
 * Returns true if the traversal is empty.
 */
bool DFS::empty() const {
  /** @todo [Part 1] */
  if(path_.empty()){return true;} else { return false; }
}

void DFS::generatePath(Point curr){
  isVisited_[curr.y*png_.width()+curr.x] = true;
  if( !(curr==start_) ){ path_.push(curr); }
  // four neighbors: right below left upper
  int move[4][2] = {{1,0},{0,1},{-1,0},{0,-1}};
  for( int i = 0; i < 4; i++ ){
    Point neighbor = Point(curr.x+move[i][0],curr.y+move[i][1]);
    if( neighbor.x < png_.width() && neighbor.y < png_.height() ){
      if( isVisited_[neighbor.y*png_.width()+neighbor.x] == false ){
        if( calculateDelta(*(png_.getPixel(neighbor.x,neighbor.y)),
            *(png_.getPixel(start_.x,start_.y))) < tolerance_ ){
          add(neighbor);
        }
      }
    }
  }
  // update path
  while( isVisited_[curr.y*png_.width()+curr.x] == true ){
    if( stack_.empty() ){ return; }
    curr = stack_.top();
    stack_.pop();
  }
  generatePath(curr);
}