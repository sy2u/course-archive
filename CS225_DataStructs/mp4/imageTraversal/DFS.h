#ifndef DFS_H
#define DFS_H

#include <iterator>
#include <cmath>
#include <list>
#include <stack>
#include <queue>

#include "../cs225/PNG.h"
#include "../Point.h"

#include "ImageTraversal.h"

using namespace cs225;

/**
 * A depth-first ImageTraversal.
 */
class DFS : public ImageTraversal {
public:
  DFS(const PNG & png, const Point & start, double tolerance);
  
  ImageTraversal::Iterator begin();
  ImageTraversal::Iterator end();

  void add(const Point & point);
  Point pop();
  Point peek() const;
  bool empty() const;

private:
  PNG png_;
  Point start_;
  double tolerance_;
  std::stack<Point> stack_;
  std::queue<Point> path_;
  std::vector<bool> isVisited_;
  void generatePath(Point curr);
};

#endif
