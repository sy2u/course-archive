#include "../cs225/HSLAPixel.h"
#include "../Point.h"

#include "ColorPicker.h"
#include "MyColorPicker.h"

using namespace cs225;

/**
 * Picks the color for pixel (x, y).
 */
HSLAPixel MyColorPicker::getColor(unsigned x, unsigned y) {
  /* @todo [Part 3] */
  HSLAPixel color;
  double delta,sum, rate;
  ((double)x-(double)y > 0) ? (delta=(double)x-(double)y) : (delta=-(double)x+(double)y) ;
  ((double)x+(double)y > 0) ? (sum=(double)x+(double)y) : (sum=-(double)x-(double)y) ;
  ( delta < sum ) ? ( rate=delta/sum ) : ( rate=sum/delta );
  color.h = 30+(210-30)*rate;
  color.s = 0.5;
  color.l = 0.5;
  color.a = 1;
  return color;
}