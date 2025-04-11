#include "StickerSheet.h"
using namespace cs225;

StickerSheet::StickerSheet(const Image &picture, unsigned max):base(picture),num(max){
    scene = new Sticker[num];
    for(unsigned i = 0; i < num; i++){
        scene[i].image = nullptr;
    }
}

StickerSheet::~StickerSheet(){
    delete[] scene;
    scene = nullptr;
}

void StickerSheet::copy(const StickerSheet &other){
    base = other.base;
    for(unsigned i = 0; i < num; i++){
        scene[i] = other.scene[i];
    }
}

StickerSheet::StickerSheet(const StickerSheet &other):num(other.num){
    scene = new Sticker[num];
    copy(other);
}

const StickerSheet & StickerSheet::operator= (const StickerSheet &other){
    delete[] scene;
    num = other.num;
    scene = new Sticker[num];
    copy(other);
    return *this;
}

void StickerSheet::changeMaxStickers(unsigned max){
    StickerSheet orig = *this;
    delete[] scene;
    scene = new Sticker[max];
    for( unsigned i = 0; (i<num) && (i<max); i++ ){
        scene[i] = orig.scene[i];
    }
    num = max;
}

int StickerSheet::addSticker(Image &sticker, unsigned x, unsigned y){
    for( unsigned i = 0; i < num; i++ ){
        if( scene[i].image == nullptr ){
            scene[i].image = &sticker;
            scene[i].x = x;
            scene[i].y = y;
            return (int)i;
        }
    }
    return -1;
}

bool StickerSheet::translate (unsigned index, unsigned x, unsigned y){
    if( scene[index].image == NULL ){ return false; }
    scene[index].x = x;
    scene[index].y = y;
    return true;
}

void StickerSheet::removeSticker(unsigned index){
    scene[index].image = nullptr;
}

Image * StickerSheet::getSticker(unsigned index) const{
    if( scene[index].image != nullptr ){ return scene[index].image; }
    return NULL;
}

Image StickerSheet::render() const{
    Image picture = base;
    for( unsigned i = 0; i < num; i++ ){
        if( scene[i].image == nullptr ){ continue; }
        Image sticker = *scene[i].image;
        if(scene[i].x+sticker.width() >= picture.width()){
            picture.resize(scene[i].x+sticker.width(),picture.height());}
        if(scene[i].y+sticker.height() >= picture.height()){ 
            picture.resize(picture.width(),scene[i].y+sticker.height());}
        for( unsigned x = 0; x < sticker.width(); x++ ){
            for( unsigned y = 0; y < sticker.height(); y++ ){
                HSLAPixel* base_ptr = picture.getPixel(scene[i].x+x,scene[i].y+y);
                const HSLAPixel* sticker_ptr = sticker.getPixel(x,y);
                if( sticker_ptr->a == 0 ){ continue; }
                *base_ptr = *sticker_ptr;
            }
        }
    }
    return picture;
}