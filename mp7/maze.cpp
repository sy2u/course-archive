/**
 * @file maze.cpp
 * Definition of SquareMaze class, including a randomly generated maze
 * and it solution.
 *
 * @author Yu Siying
 * @date Summer 2023
 */

#include "maze.h"
#include "dsets.h"
#include <algorithm>
#include <random>

/**
 * Default constructor, creat an empty maze.
 **/
SquareMaze::SquareMaze(){
    width_ = 0;
}

/**
 * Makes a new SquareMaze of the given height and width.
 * @param width -- The width of the SquareMaze (number of cells).
 * @param height -- The height of the SquareMaze (number of cells).
 **/
void SquareMaze::makeMaze(int width, int height){
    int cellNum = width*height;
    width_ = width; height_ = height;
    // Initialize isolated cells in the maze
    DisjointSets connect;
    connect.addelements(cellNum);
    // Generate Wall ID Set and Initial Maze
    vector<vector<int>> wall;
    wall.reserve(2*cellNum);
    for(int x = 0; x < width; x++){
        for( int y = 0; y < height; y++ ){
            maze.push_back(Cell());
            if(x!=width-1){
                vector<int> pos = {x,y,0};
                wall.push_back(pos); }
            if(y!=height-1){ 
                vector<int> pos = {x,y,1};
                wall.push_back(pos); }
        }
    }
    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::shuffle(wall.begin(),wall.end(),generator);

    // Loop until the maze is fully connected
    while(connect.size(0)!=cellNum){
        int x = wall.back()[0];
        int y = wall.back()[1];
        int dir = wall.back()[2];
        wall.pop_back();
        // Check whether a cycle would occur
        int cellid1 = cellID(x,y);
        int cellid2; if( dir==0 ){ cellid2=cellID(x+1,y); } else { cellid2=cellID(x,y+1); }
        if( connect.find(cellid1) == connect.find(cellid2) ){
            continue;
        } else { // No cycle, this wall is moveable
            setWall(x,y,dir,false);
            connect.setunion(cellid1,cellid2);
        }
    }
}

/**
 * Determine whether it is possible to travel in the given direction 
 * from the square at coordinates (x,y).
 * @param x -- x coordinate
 * @param y -- y coordinate
 * @param dir -- 0: rightward; 1: downward; 2: leftward; 3: upward
 * @return Whether it's possible to travel in the specified direction 
 **/
bool SquareMaze::canTravel(int x, int y, int dir) const{
    if( (x==0&&dir==2)||(y==0&&dir==3) ){ return false; }
    switch(dir){
    case 0: return !maze[cellID(x,y)].right;
    case 1: return !maze[cellID(x,y)].down;
    case 2: return !maze[cellID(x-1,y)].right;
    case 3: return !maze[cellID(x,y-1)].down;
    default: return false;
    }
}

/**
 * Sets whether or not the specified wall exists. 
 * Can't prevent cycle. Can't protect the boundary of the maze.
 * Parameter: x -- x coordinate.
 *            y -- y coordinate.
 *            dir -- 0: rightward; 1: downward; 2: leftward; 3: upward.
 *            exists -- true if setting the wall to exist, false otherwise. 
 **/
void SquareMaze::setWall(int x, int y, int dir, bool exists){
    switch(dir){
    case 0: maze[cellID(x,y)].right = exists;   break;
    case 1: maze[cellID(x,y)].down = exists;    break;
    case 2: maze[cellID(x-1,y)].right = exists; break;
    case 3: maze[cellID(x,y-1)].down = exists;  break;
    default: break;
    }
}

/**
 * Solve the maze. Select the square in the bottom row with the largest distance (num 
 * of cells) from the origin as the destination of the maze. If multiple paths of maximum 
 * length exist, use the one with the destination cell that has the smallest x value.
 * @return The winning path from the origin to the destination as a vector of integers, 
 *         where each integer represents the direction of a step.
 *         0: rightward; 1: downward; 2: leftward; 3: upward.
 **/
vector<int> SquareMaze::solveMaze(){
    vector<int> temp, opti;
    int optiDist = 0; int count = 0; int optiX = width_;
    solveHelper(0,0,-1,0,optiDist,temp,opti,optiX,count);
    optiX_ = optiX;
    return opti;
}

/**
 * Recursively implement DFS to solve the maze.
 * @param x -- x coordinate
 * @param y -- y coordinate
 * @param dir -- 0: rightward; 1: downward; 2: leftward; 3: upward.
 * @param soln -- the winning path from the origin to the destination as a vector of integers
 **/
void SquareMaze::solveHelper(int x, int y, int prevDir, int currDist, int& optiDist,
                            vector<int>& temp, vector<int>& opti, int& optiX, int& count){
    if( y==height_-1 ){
        // Base case: Arrive at the end
        count++;
        if( (currDist>optiDist) || (currDist==optiDist)&&(x<optiX) ){
            opti = temp;
            optiDist = currDist;
            optiX = x;
        }
        if( count == width_ ){ return; }
    }
    // Go right
    if( prevDir!=2 && canTravel(x,y,0) ){ 
        temp.push_back(0);
        solveHelper(x+1,y,0,currDist+1,optiDist,temp,opti,optiX,count);
        if( count == width_ ){ return; }
        temp.pop_back();}
    // Go down
    if( prevDir!=3 && canTravel(x,y,1) ){ 
        temp.push_back(1);
        solveHelper(x,y+1,1,currDist+1,optiDist,temp,opti,optiX,count);
        if( count == width_ ){ return; }
        temp.pop_back();}
    // Go left
    if( prevDir!=0 && canTravel(x,y,2) ){ 
        temp.push_back(2);
        solveHelper(x-1,y,2,currDist+1,optiDist,temp,opti,optiX,count);
        if( count == width_ ){ return; }
        temp.pop_back();}
    // Go up
    if( prevDir!=1 && canTravel(x,y,3) ){ 
        temp.push_back(3);
        solveHelper(x,y-1,3,currDist+1,optiDist,temp,opti,optiX,count);
        if( count == width_ ){ return; }
        temp.pop_back();}
}

/**
 * Draws the maze without the solution.
 * @return A PNG of the unsolved SquareMaze.
 **/
PNG* SquareMaze::drawMaze() const{

    PNG* png = new PNG(width_*10+1,height_*10+1);
    HSLAPixel black = HSLAPixel(0,0,0,1);

    for(int x = 0; x < width_*10+1; x++){ 
        if( x>=10 ){ *png->getPixel(x,0) = black; }
        *png->getPixel(x,height_*10) = black;
    }
    for(int y = 0; y < height_*10+1; y++){
        *png->getPixel(0,y) = black;
        *png->getPixel(width_*10,y) = black;
    }

    for( int x = 0; x < width_; x++ ){
        for( int y = 0; y < height_; y++ ){
            if( maze[cellID(x,y)].right ){ 
                for( int k = 0; k < 11; k++ ){
                    *png->getPixel((x+1)*10,y*10+k) = black; }
                }
            if( maze[cellID(x,y)].down ){
                for( int k = 0; k < 11; k++ ){
                    *png->getPixel(x*10+k, (y+1)*10) = black; }
                }
            }
        }
    
    return png;
}

/**
 * Modifies the PNG from drawMaze to show the solution vector and the exit.
 * @return A PNG of the unsolved SquareMaze.
 **/
PNG* SquareMaze::drawMazeWithSolution(){
    PNG* png = drawMaze();
    vector<int> soln = solveMaze();
    int currPixel[2] = {5,5};
    HSLAPixel red = HSLAPixel(0,1,0.5,1);
    for( int i = 0; i < soln.size(); i++ ){
        switch(soln[i]){
        case 0: //right
            for( int j = 0; j < 10; j++ ){ *png->getPixel(currPixel[0]++,currPixel[1])= red; }  break;
        case 1: //down
            for( int j = 0; j < 10; j++ ){ *png->getPixel(currPixel[0],currPixel[1]++)= red; }  break;
        case 2: //left
            for( int j = 0; j < 10; j++ ){ *png->getPixel(currPixel[0]--,currPixel[1])= red; }  break;
        case 3: //up
            for( int j = 0; j < 10; j++ ){ *png->getPixel(currPixel[0],currPixel[1]--)= red; }  break;
        default: break;
        }
    }
    *png->getPixel(currPixel[0],currPixel[1]) = red;

    for( int k = 1; k < 10; k++ ){
        *png->getPixel(optiX_*10+k,height_*10) = HSLAPixel(0,0,1,1);
    }

    return png;

}
