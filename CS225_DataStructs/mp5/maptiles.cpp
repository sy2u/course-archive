/**
 * @file maptiles.cpp
 * Code for the maptiles function.
 */

#include <iostream>
#include <map>
#include "maptiles.h"

using namespace std;

Point<3> convertToLAB(HSLAPixel pixel) {
    Point<3> result(pixel.h/360, pixel.s, pixel.l);
    return result;
}

MosaicCanvas* mapTiles(SourceImage const& theSource,
                       vector<TileImage>& theTiles)
{
    MosaicCanvas* mosaic = new MosaicCanvas(theSource.getRows(),theSource.getColumns());

    // Construct point->image map & point KDTree
    std::vector<Point<3>> averageColor;
    std::map<Point<3>,int> imgMap;
    for( int i = 0; i < int(theTiles.size()); i++ ){
        HSLAPixel hsl_color = theTiles[i].getAverageColor();
        Point<3> point_color = Point<3>(hsl_color.h/360,hsl_color.s,hsl_color.l);
        averageColor.push_back(point_color);
        imgMap.insert(pair<Point<3>,int>(point_color,i));
    }
    KDTree<3> kdtree = KDTree<3>(averageColor);

    // Traverse regions in SourceImage and set tile for mosaic
    for( int x = 0; x < theSource.getRows(); x++ ){
        for( int y = 0; y < theSource.getColumns(); y++ ){
            HSLAPixel orig_color = theSource.getRegionColor(x,y);
            Point<3> orig_point = Point<3>(orig_color.h/360,orig_color.s,orig_color.l);
            Point<3> tile_point = kdtree.findNearestNeighbor(orig_point);
            std::map<Point<3>,int>::iterator it = imgMap.find(tile_point);
            if( it==imgMap.end() ){ return NULL;}
            TileImage* img = &theTiles[it->second];
            mosaic->setTile(x,y,img);
        }
    }

    return mosaic;
}

TileImage* get_match_at_idx(const KDTree<3>& tree,
                                  map<Point<3>, int> tile_avg_map,
                                  vector<TileImage>& theTiles,
                                  const SourceImage& theSource, int row,
                                  int col)
{
    // Create a tile which accurately represents the source region we'll be
    // using
    HSLAPixel avg = theSource.getRegionColor(row, col);
    Point<3> avgPoint = convertToLAB(avg);
    Point<3> nearestPoint = tree.findNearestNeighbor(avgPoint);

    // Check to ensure the point exists in the map
    map< Point<3>, int >::iterator it = tile_avg_map.find(nearestPoint);
    if (it == tile_avg_map.end())
        cerr << "Didn't find " << avgPoint << " / " << nearestPoint << endl;

    // Find the index
    int index = tile_avg_map[nearestPoint];
    return &theTiles[index];

}
