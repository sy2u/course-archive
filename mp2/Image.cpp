#include "Image.h"
using namespace cs225;

void Image::lighten(){
  for( unsigned int x = 0; x < width(); x++ ){
    for( unsigned int y = 0; y < height(); y++ ){
      HSLAPixel* ptr = getPixel(x,y);
      (ptr->l < 0.9) ? (ptr->l += 0.1) : (ptr->l = 1);
    }
  }
}

void Image::lighten(double amount){
  for( unsigned int x = 0; x < width(); x++ ){
    for( unsigned int y = 0; y < height(); y++ ){
      HSLAPixel* ptr = getPixel(x,y);
      (ptr->l < 1-amount) ? (ptr->l += amount) : (ptr->l = 1);
    }
  }
}

void Image::darken(){
  for( unsigned int x = 0; x < width(); x++ ){
    for( unsigned int y = 0; y < height(); y++ ){
      HSLAPixel* ptr = getPixel(x,y);
      (ptr->l > 0.1) ? (ptr->l -= 0.1) : (ptr->l = 0);
    }
  }
}

void Image::darken(double amount){
  for( unsigned int x = 0; x < width(); x++ ){
    for( unsigned int y = 0; y < height(); y++ ){
      HSLAPixel* ptr = getPixel(x,y);
      (ptr->l > amount) ? (ptr->l -= amount) : (ptr->l = 0);
    }
  }
}

void Image::saturate(){
  for( unsigned int x = 0; x < width(); x++ ){
    for( unsigned int y = 0; y < height(); y++ ){
      HSLAPixel* ptr = getPixel(x,y);
      (ptr->s < 0.9) ? (ptr->s += 0.1) : (ptr->s = 1);
    }
  }
}

void Image::saturate(double amount){
  for( unsigned int x = 0; x < width(); x++ ){
    for( unsigned int y = 0; y < height(); y++ ){
      HSLAPixel* ptr = getPixel(x,y);
      (ptr->s < 1-amount) ? (ptr->s += amount) : (ptr->s = 1);
    }
  }
}

void Image::desaturate(){
  for( unsigned int x = 0; x < width(); x++ ){
    for( unsigned int y = 0; y < height(); y++ ){
      HSLAPixel* ptr = getPixel(x,y);
      (ptr->s > 0.1) ? (ptr->s -= 0.1) : (ptr->s = 0);
    }
  }
}

void Image::desaturate(double amount){
  for( unsigned int x = 0; x < width(); x++ ){
    for( unsigned int y = 0; y < height(); y++ ){
      HSLAPixel* ptr = getPixel(x,y);
      (ptr->s > amount) ? (ptr->s -= amount) : (ptr->s = 0);
    }
  }
}

void Image::grayscale(){
  for( unsigned int x = 0; x < width(); x++ ){
    for( unsigned int y = 0; y < height(); y++ ){
      HSLAPixel* ptr = getPixel(x,y);
      ptr->s = 0;
    }
  }
}

void Image::rotateColor(double degrees){
  for( unsigned int x = 0; x < width(); x++ ){
    for( unsigned int y = 0; y < height(); y++ ){
      HSLAPixel* ptr = getPixel(x,y);
      ptr->h += degrees;
      if( ptr->h > 360 ){ ptr->h -= 360; }
      if( ptr->h < 0 ){ ptr->h += 360; }
    }
  }
}

void Image::illinify(){
  for( unsigned int x = 0; x < width(); x++ ){
    for( unsigned int y = 0; y < height(); y++ ){
      HSLAPixel* ptr = getPixel(x,y);
      (113<ptr->h && ptr->h<293) ? (ptr->h = 216) : (ptr->h = 11);
    }
  }
}

void Image::scale(double factor){
  Image OrigImage(*this);
  resize(width()*factor,height()*factor);
  for( unsigned int x = 0; x < width(); x++ ){
    for( unsigned int y = 0; y < height(); y++ ){
      HSLAPixel* NewPtr = getPixel(x,y);
      HSLAPixel* OrigPtr = OrigImage.getPixel((int)x/factor,(int)y/factor);
      *NewPtr = *OrigPtr;
    }
  }
}

void Image::scale(unsigned w, unsigned h){
  double factor_w = (double)w/(double)width();
  double factor_h = (double)h/(double)height();
  Image OrigImage(*this);
  resize(w,h);
  for( unsigned int x = 0; x < width(); x++ ){
    for( unsigned int y = 0; y < height(); y++ ){
      HSLAPixel* NewPtr = getPixel(x,y);
      HSLAPixel* OrigPtr = OrigImage.getPixel((int)x/factor_w,(int)y/factor_h);
      *NewPtr = *OrigPtr;
    }
  }
}