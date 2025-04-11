#ifndef HSLAPIXEL_H
#define HSLAPIXEL_H

namespace cs225{
    class HSLAPixel{
        public:
            /**
                 * Constructs a default HSLAPixel.
                 * A default pixel is completely opaque (non-transparent) and white. 
                 * Opaque implies that the alpha component of the pixel is 1.0. Lower values are transparent. 
                 */
            HSLAPixel();
            /**
                 * Constructs an opaque HSLAPixel with the given hue, saturation, and luminance values. 
                 * The alpha component of the pixel constructed should be 1.0. 
                 * @param hue - Hue value for the new pixel, in degrees [0, 360]. 
                 * @param saturation - Saturation value for the new pixel, [0, 1]. 
                 * @param luminance - Luminance value for the new pixel, [0, 1]. 
                 */ 
            HSLAPixel(double hue, double saturation, double luminance);
            /**
                 * Constructs an opaque HSLAPixel with the given hue, saturation, luminance, and alpha values. 
                 * @param hue - 	Hue value for the new pixel, in degrees [0, 360]. 
                 * @param saturation - Saturation value for the new pixel, [0, 1]. 
                 * @param luminance - Luminance value for the new pixel, [0, 1]. 
                 * @param alpha - Alpha value for the new pixel, [0, 1]. 
                 */ 
            HSLAPixel(double hue, double saturation, double luminance, double alpha);

            double h;       /*Double for the hue of the pixel, in degrees [0, 360]. */
            double s;       /*Double for the saturation of the pixel, [0, 1]. */
            double l;       /*Double for the luminance of the pixel, [0, 1]. */
            double a;       /*Double for the alpha of the pixel, [0, 1]. */

    };
}

#endif