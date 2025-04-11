
#include "cs225/PNG.h"
#include "FloodFilledImage.h"
#include "Animation.h"

#include "imageTraversal/DFS.h"
#include "imageTraversal/BFS.h"

#include "colorPicker/RainbowColorPicker.h"
#include "colorPicker/GradientColorPicker.h"
#include "colorPicker/GridColorPicker.h"
#include "colorPicker/SolidColorPicker.h"
#include "colorPicker/MyColorPicker.h"

using namespace cs225;

int main() {

  // @todo [Part 3]
  // - The code below assumes you have an Animation called `animation`
  // - The code provided below produces the `myFloodFill.png` file you must
  //   submit Part 3 of this assignment -- uncomment it when you're ready.
  PNG png;
  png.readFromFile("pacman.png");
  FloodFilledImage image(png);
  DFS dfs(png, Point(10, 10), 0.2);
  HSLAPixel color(231, 1, 0.5);
  SolidColorPicker solid(color);
  image.addFloodFill( dfs, solid );
  BFS bfs(png, Point(100, 150), 0.18);
  MyColorPicker random;
  image.addFloodFill( bfs, random );

  Animation animation = image.animate(1000);

  PNG Frame = animation.getFrame( animation.frameCount() - 1 );
  Frame.writeToFile("result.png");

  return 0;
}
