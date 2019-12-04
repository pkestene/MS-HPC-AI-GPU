/*
 * g++ -Wall -Wextra -o test test.cpp -lmgl
 */

#include <mgl2/mgl.h>
int main()
{
    mglGraph *gr = new mglGraph;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // put sample code here
    mglData a(50,40);
    a.Modify("0.6*sin(2*pi*x)*sin(3*pi*y) + 0.4*cos(3*pi*(x*y))");
    
    gr->Rotate(40,60);
    gr->Light(true);
    gr->Box();
    gr->Surf(a);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //gr->ShowImage();
    gr->WritePNG("test.png"); 
    delete gr;
    return 0;
}
