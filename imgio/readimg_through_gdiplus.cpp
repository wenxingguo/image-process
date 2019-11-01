#include <string>
#include <windows.h>
#include <gdiplus.h>
#include <FL/Fl_Window.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_RGB_Image.H>

using namespace std;
//using namespace Gdiplus;

int main() {
    Gdiplus::GdiplusStartupInput gdiplusstartupinput;
    ULONG_PTR gdiplustoken;
    Gdiplus::GdiplusStartup(&gdiplustoken, &gdiplusstartupinput, nullptr);

    wstring infilename(L"../img/cat.png");
    //string outfilename("color.txt");

    Gdiplus::Bitmap* bmp = new Gdiplus::Bitmap(infilename.c_str());
    UINT height = bmp->GetHeight();
    UINT width  = bmp->GetWidth();
    //cout << "width " << width << ", height " << height << endl;

    Gdiplus::Color color;
    //ofstream fout(outfilename.c_str());
    unsigned char data[width*height*3];
    int i = 0;
    for (UINT y = 0; y < height; y++){
    for (UINT x = 0; x < width ; x++) {
            bmp->GetPixel(x, y, &color);
            data[i*3] = color.GetRed();
            data[i*3+1] = color.GetGreen();
            data[i*3+2] = color.GetBlue();
            ++i;
            //fout << x << "," << y << ";"
                // << (int)color.GetRed()   << ","
                // << (int)color.GetGreen() << ","
                // << (int)color.GetBlue()  << endl;
        }
    }

    //fout.close();

    delete bmp;
    Gdiplus::GdiplusShutdown(gdiplustoken);

    Fl_Window w(width, height);
    Fl_Box b(0,0,width, height);
    Fl_RGB_Image img(data, width, height, 3);
    b.image(img);
    w.end();
    w.show();
    return Fl::run();
}
