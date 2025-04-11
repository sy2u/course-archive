/**
 * @file maze.h
 * Definition of SquareMaze class, including a randomly generated maze
 * and it solution.
 *
 * @author Yu Siying
 * @date Summer 2023
 */

#ifndef MAZE_H
#define MAZE_H

#include <vector>
#include "cs225/PNG.h"

using std::vector;
using namespace cs225;

/**
 * Cell Strcture.
 * Definition of single cell of the maze. Including the "down" and "right" walls.
 **/
struct Cell{
    bool down;
    bool right;
    Cell():down(true),right(true){}; // all walls are defaulted to be exist
};

/**
 * SquareMaze Class.
 * Each SquareMaze object represents a randomly-generated square maze 
 * and its solution. All cells of the maze are squares.
 **/
class SquareMaze{
    private:
        int width_;
        int height_;
        int optiX_;
        vector<Cell> maze;
        /**
         * Helper function. Return the index of specific cell in maze vector.
         **/
        int cellID(int x, int y)const{ return x+y*width_; }
        /**
         * Helper function. Recursively implement DFS to solve the maze.
         **/
        void solveHelper(int x, int y, int prevDir, int currDist, int& optiDist,
                        vector<int>& temp, vector<int>& opti, int& optiX, int& count);
    public:
        /**
         * Default constructor, creat an empty maze.
         **/
        SquareMaze();
        /**
         * Makes a new SquareMaze of the given height and width.
         * Parameter: width -- The width of the SquareMaze (number of cells).
         *            height -- The height of the SquareMaze (number of cells).
         **/
        void makeMaze(int width, int height);
        /**
         * Determine whether it is possible to travel in the given direction 
         * from the square at coordinates (x,y).
         * Parameter: x -- x coordinate.
         *            y -- y coordinate.
         *            dir -- 0: rightward; 1: downward; 2: leftward; 3: upward.
         * Return: Whether it's possible to travel in the specified direction.
         **/
        bool canTravel(int x, int y, int dir) const;
        /**
         * Sets whether or not the specified wall exists. Can't prevent cycle.
         * Parameter: x -- x coordinate.
         *            y -- y coordinate.
         *            dir -- 0: rightward; 1: downward; 2: leftward; 3: upward.
         *            exists -- true if setting the wall to exist, false otherwise. 
         **/
        void setWall(int x, int y, int dir, bool exists);
        /**
         * Solve the maze. Select the square in the bottom row with the largest distance (num 
         * of cells) from the origin as the destination of the maze. If multiple paths of maximum 
         * length exist, use the one with the destination cell that has the smallest x value.
         * Return: The winning path from the origin to the destination as a vector of integers, 
         *         where each integer represents the direction of a step.
         *         0: rightward; 1: downward; 2: leftward; 3: upward.
         **/
        vector<int> solveMaze();
        /**
         * Draws the maze without the solution.
         * Return: A PNG of the unsolved SquareMaze.
         **/
        PNG* drawMaze() const;
        /**
         * Modifies the PNG from drawMaze to show the solution vector and the exit.
         * Return: A PNG of the unsolved SquareMaze.
         **/
        PNG* drawMazeWithSolution();
};

#endif