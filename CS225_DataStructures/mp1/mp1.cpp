#include <string>
#include "mp1.h"
#include "cs225/PNG.h"
#include "cs225/HSLAPixel.h"

using namespace cs225;

void rotate(std::string inputFile, std::string outputFile) {
    PNG origin,rotate;
    origin.readFromFile(inputFile);
    rotate = origin;
    for (unsigned x = 0; x < origin.width(); x++) {
        for (unsigned y = 0; y < origin.height(); y++) {
            HSLAPixel* ptr_ori = origin.getPixel(x,y); 
            HSLAPixel* ptr_rot = rotate.getPixel(origin.width()-x-1,origin.height()-y-1); 
            ptr_rot->h = ptr_ori->h;
            ptr_rot->s = ptr_ori->s;
            ptr_rot->l = ptr_ori->l;
            ptr_rot->a = ptr_ori->a;
        }
    }
    rotate.writeToFile(outputFile);
}