/**
 * @file testsquaremaze.cpp
 * Performs basic tests of SquareMaze.
 * @date April 2007
 * @author Jonathan Ray
 */

#include <iostream>
#include "dsets.h"
#include "maze.h"
#include "cs225/PNG.h"

using namespace std;

int main()
{
    SquareMaze m;
    m.makeMaze(50, 50);
    std::cout << "MakeMaze complete" << std::endl;

    PNG* unsolved = m.drawMaze();
    unsolved->writeToFile("unsolved.png");
    delete unsolved;
    std::cout << "drawMaze complete" << std::endl;

    std::vector<int> sol = m.solveMaze();
    std::cout << "solveMaze complete" << std::endl;

    PNG* solved = m.drawMazeWithSolution();
    solved->writeToFile("solved.png");
    delete solved;
    std::cout << "drawMazeWithSolution complete" << std::endl;

    // PNG right,wrong;
    // wrong.readFromFile("testDrawSolutionMed.png");
    // right.readFromFile("tests/soln_testDrawSolutionMed.png");
    // std::cout<<right.width()<<"&"<<wrong.width()<<std::endl;
    // std::cout<<right.height()<<"&"<<wrong.height()<<std::endl;
    // for(unsigned x = 0; x < right.width(); x++){
    //     for(unsigned y = 0; y < right.height(); y++){
    //     if( *(right.getPixel(x,y)) != *(wrong.getPixel(x,y)) ){
    //         std::cout<<"x:"<<x<<" y:"<<y<<" color:"<<*(right.getPixel(x,y))<<"&"<<*wrong.getPixel(x,y)<<std::endl;
    //     }
    //     }
    // }

    return 0;
}
