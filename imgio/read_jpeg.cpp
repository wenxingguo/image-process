#include "stdio.h"
#include "jpeglib.h"
#include <iostream>
#include "FL/Fl_Window.H"
#include "FL/Fl_Box.H"
#include "FL/Fl_RGB_Image.H"


int main(int argv, char* argc[]){
    struct jpeg_decompress_struct cinof;
    struct jpeg_error_mgr jerr;
    cinof.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinof);
    FILE* infile = fopen(argc[1], "rb");
    if(infile == nullptr){
        std::cout << "open faild!!" << std::endl;
        return -1;
    }
    jpeg_stdio_src(&cinof, infile);
    jpeg_read_header(&cinof, TRUE);
    int width  = cinof.image_width, height = cinof.image_height, channel = cinof.num_components;
    //std::cout << width << std::endl;
    unsigned char* data = new unsigned char[width*height*channel];
    jpeg_start_decompress(&cinof);
    JSAMPROW row_pointer[1];
    while(cinof.output_scanline < cinof.output_height){
        //std::cout << cinof.output_scanline << std::endl;
        row_pointer[0] = &data[cinof.output_scanline*cinof.image_width*cinof.num_components];
        jpeg_read_scanlines(&cinof, row_pointer, 1);
    }
    jpeg_finish_decompress(&cinof);
    jpeg_destroy_decompress(&cinof);
    fclose(infile);
    Fl_Window w(width, height);
    Fl_Box b(0 ,0, width, height);
    Fl_RGB_Image img(data, width, height, 3);
    b.image(img);
    w.end();
    w.show();
    return Fl::run();
}