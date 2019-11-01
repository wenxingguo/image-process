#include "stdio.h"
#include "png.h"
#include <iostream>
#include "FL/Fl_Window.H"
#include "FL/Fl_Box.H"
#include "FL/Fl_RGB_Image.H"


int main(int argc, char* argv[]){
    png_structp png_ptr;
    png_infop info_ptr;
    unsigned int width, height;
    int bit_depth, channels, color_type;

    FILE* infile;
    infile = fopen(argv[1], "rb");
    if(infile == nullptr){
        std::cout << "open fail!!" << std::endl;
        return -1;
    }
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if(png_ptr == NULL){
        fclose(infile);
        return -1;
    }
    info_ptr = png_create_info_struct(png_ptr);
    if(info_ptr == NULL){
        fclose(infile);
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        return -1;
    }
    if(setjmp(png_jmpbuf(png_ptr))){
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        fclose(infile);
        return -1;
    }

    png_init_io(png_ptr, infile);
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_EXPAND, 0);

    png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type,
                    nullptr, nullptr, nullptr);
    png_bytepp png_row_infop = png_get_rows(png_ptr, info_ptr);
    unsigned int bufsize = 0;
    if(color_type == PNG_COLOR_TYPE_RGB){
        channels = 3;
    }else if(color_type == PNG_COLOR_TYPE_RGBA){
        channels = 4;
    }else{
        return 0;
    }
    //std::cout << channels << std::endl;
    unsigned char* data = new unsigned char[width*height*channels](); //注意栈溢出
    for (int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            for(int ch = 0; ch < channels; ++ch){
                data[(i*width+j)*channels +  ch] = png_row_infop[i][j*channels+ch];
            }
        }
    }

    Fl_Window w(width, height);
    Fl_Box b(0,0, width, height);
    Fl_RGB_Image img(data, width, height, channels);
    b.image(img);

    w.end();
    w.show();
    return Fl::run();
}