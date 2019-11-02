#ifndef BASE_H
#define BASE_H
#define Base_assert(x) assert(x) //判断越界

#include <vector>

typedef unsigned int UINT;
typedef std::vector<UINT> DOMAIN; //[x1, y1, x2, y2]

template<class DataType>
class Base{
    typedef DataType* DataTypep;
    protected:
    UINT width;
    UINT height;
    UINT channels;
    DataTypep data;
    
    public: 
    Base(DataTypep in_data, int in_width, int in_height, int in_channals);

    Base(int in_width, int in_height, int in_channels);

    Base(const Base<DataType>& other_Base);

    DataType& operator()(UINT x, UINT y, UINT channel);

    DataType& operator()(UINT x, UINT y, UINT channel) const; //访问常量对象;
    
    UINT get_width() const{
        return width;
    }

    void set_width(UINT in_width){
        width = in_width;
    }
    
    UINT get_height() const{
        return height
    }

    void set_height(int in_height) {
        height = in_height;
    }
    
    UINT get_channels() const{
        return channels;
    }

    void set_channels(int in_channels){
        channels = in_channels;
    }
    
    DataTypep get_data_handle() const{
        return data;
    }
    
    ~Base(){
        if(data){
            delete data;
            data = nullptr;
        }
    }

    void flip(); //反转

    void stick(const Base<DataType>& sub_Base, DOMAIN domain);

    void diff(const Base<DataType>& Base2, baas<DataType>& outBase);//比较两个Base的不同，作差

    void sub_Base(baas<DataType>& subBase, DOMAIN domain); //获得一个Base的子图






};

template<class DataType>
Base<DataType>::Base(DataTypep in_data, int in_width, int in_height, int in_channels)
:data(in_data),width(in_width),height(in_height),channels(in_channels){

}

template<class DataType>
Base<DataType>::Base(int in_width, int in_height, int in_channels):width(in_width),height(in_height),
channels(in_channels){
    data = new DataType[width*height*channels];
}

template<class DataType>
Base<DataType>::Base(const Base<DataType>& other_base):width(other_base.get_width()),height(other_base.get_height()),
channels(other_base.get_channels()){
    //DataTypep temp_data = other_base.get_data_handle();
    data = new DataType[width*height*channels]; //申请内存
    memcpy(data, other_base.get_data_handle(), sizeof(DataType)*width*height*channels); //直接拷贝内存
    /*for (int x = 1; x <= width; ++x){
        for(int y = 1; x <= height; ++y){
            for(int j = 0; j < channels; ++j){
                (*this)(x,y,j) = other_base(x,y,j);
            }
        }
    }*/
    
}




#endif