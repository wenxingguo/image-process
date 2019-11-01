#include "stdio.h"
#include "jpeglib.h"
#include <iostream>
#include <FL/Fl_Window.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_RGB_Image.H>

int main(int argc, char* argv[]){
    unsigned char data[512*512*3];
    for(int i = 0; i < 512*512*3; i += 3){
        data[i] = 255;
        data[i+1] = 255;
        data[i+2] = 255;
        //std::cout << i << std::endl;
    }
    //data[100] = 255;
    std::cout << (int)data[100] << " " << (int)data[101] << " " <<(int)data[102] << std::endl;
    struct jpeg_compress_struct jcs;
    struct jpeg_error_mgr jem;
    jcs.err = jpeg_std_error(&jem);

    jpeg_create_compress(&jcs);
    
    FILE* outfile = fopen("./test.jpeg", "wb");
    if(outfile == nullptr){
        std::cout << " open fail" << std::endl;
        return -1;
    }

    jpeg_stdio_dest(&jcs, outfile);
    jcs.image_width = 512;
    jcs.image_height = 512;
    jcs.input_components = 3;
    jcs.in_color_space = JCS_RGB;
    jpeg_set_defaults(&jcs);
    //jpeg_set_quality(&jcs, 100, true);
    jpeg_start_compress(&jcs, true);
    JSAMPROW row_pointer[0];
    while(jcs.next_scanline < jcs.image_height){
        //std::cout << jcs.num_components<< std::endl;
        row_pointer[0] = &data[jcs.next_scanline*jcs.image_width*jcs.num_components];
        jpeg_write_scanlines(&jcs, row_pointer, 1);
    }
    jpeg_finish_compress(&jcs);
    jpeg_destroy_compress(&jcs);
    fclose(outfile);

    Fl_Window w(512,512);
    Fl_Box b(0,0,512,512);
    Fl_RGB_Image img(data, 512,512, 3);
    b.image(img);
    w.end();
    w.show();
    return Fl::run();
}