#ifndef FLOODFILLEDIMAGE_H
#define FLOODFILLEDIMAGE_H

#include "cs225/PNG.h"
#include <list>
#include <queue>
#include <iostream>

#include "colorPicker/ColorPicker.h"
#include "imageTraversal/ImageTraversal.h"

#include "Point.h"
#include "Animation.h"

using namespace cs225;

class FloodFilledImage {
public:
  FloodFilledImage(const PNG & png);
  void addFloodFill(ImageTraversal & traversal, ColorPicker & colorPicker);
  Animation animate(unsigned frameInterval) const;

private:
  const PNG* base_;
  struct Flood{
    ImageTraversal* image_;
    ColorPicker* color_;
    Flood(ImageTraversal* image,ColorPicker* color):image_(image),color_(color){}
  };
  std::list<Flood> floodList;
};

#endif
