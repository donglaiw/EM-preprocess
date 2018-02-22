#include "idm_main.h"

float patch_distance (int A_x,int A_y, int B_x, int B_y, int im_chan, int im_row, int im_col,
        int patch_sz, float *img1, float *img2){    
    float dist=0,temp_h;
    int c,x,y,count=0;
    /* only move around patchB */
    int pre = 0;
    for(c=0; c<im_chan; c++){
        for(y=-patch_sz; y<=patch_sz; y++)
            for(x=-patch_sz; x<=patch_sz; x++)
            {
                if((A_x + x)>=0 && (A_y + y)>=0 && (A_x + x)<im_row && (A_y + y)<im_col
                        && (B_x + x)>=0 && (B_y + y)>=0 && (B_x + x)<im_row && (B_y + y)<im_col)
                {
                    temp_h = (float)img1[pre + ((A_y + y) * im_row) + (A_x + x)] -
                            (float)img2[pre + ((B_y + y) * im_row) + (B_x + x)];
                    dist+=temp_h*temp_h;
                    count++;
                    //printf("%d,%f\n",count,dist);
                }
            }
        pre += im_row*im_col;
    }
    return dist/count;
}

void idm_dist(float *img1, float *img2, float *dis, 
        int im_chan, int im_row, int im_col, int patch_sz, int warp_sz){
    /* assume same size img */
    float best_dis;
    float temp;
    int x,y,xx,yy;
    
    /* 3) Return distance */
    int count=0;
    for (y=0; y<im_col; y++){
        for (x=0; x<im_row; x++)        
            {
                best_dis=FLT_MAX;
                for(xx=x-warp_sz; xx<=x+warp_sz; xx++){
                    for(yy=y-warp_sz; yy<=y+warp_sz; yy++){
                        if(xx >= 0 && yy >= 0 && xx < im_row && yy < im_col){
                            temp = patch_distance(x, y, xx, yy, im_chan, im_row, im_col, 
                                    patch_sz, img1, img2);
                            if(temp<best_dis){
                                best_dis = temp;                                
                            }
                        }
                    }
                }
                dis[count] = best_dis;             
                count++;
            }
        }
    return;
}
