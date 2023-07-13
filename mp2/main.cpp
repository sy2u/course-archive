#include "Image.h"
#include "StickerSheet.h"

using namespace cs225;

int main() {
  Image alma,car,flower,illini,output;
  car.readFromFile("car.png");
  alma.readFromFile("alma.png");
  flower.readFromFile("flower.png");
  illini.readFromFile("i.png");
  car.scale(1.1);
  flower.scale(1.1);
  illini.scale(0.2);

  StickerSheet picture(alma,20);
  picture.addSticker(car,585,215);
  picture.addSticker(flower,100,380);
  picture.addSticker(flower,140,420);
  picture.addSticker(flower,190,440);
  picture.addSticker(flower,240,450);
  picture.addSticker(flower,290,450);
  picture.addSticker(flower,340,450);
  picture.addSticker(flower,390,450);
  picture.addSticker(flower,440,450);
  picture.addSticker(flower,490,450);
  picture.addSticker(flower,540,440);
  picture.addSticker(illini,425,110);

  output = picture.render();
  output.resize(900,600);
  output.writeToFile("myImage.png");
  return 0;
}
