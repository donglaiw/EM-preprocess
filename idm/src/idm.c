#include <float.h>
#include <stdint.h>
float patch_distance (int A_x,int A_y, int B_x, int B_y, int im_row, int im_col,
        int patch_row, int patch_col, float *img1, float *img2){    
    float dist=0,temp_h;
    int x,y,count=0;
    /* only move around patchB */
    for(x=-patch_row; x<=patch_row; x++)
        for(y=-patch_col; y<=patch_col; y++)
        {
            if((A_x + x)>=0 && (A_y + y)>=0 && (A_x + x)<im_row && (A_y + y)<im_col
                    && (B_x + x)>=0 && (B_y + y)>=0 && (B_x + x)<im_row && (B_y + y)<im_col)
            {
                temp_h = img1[((A_y + y) * A_x_size) + (A_x + x)] -
                        img2[((B_y + y) * B_x_size) + (B_x + x)];
                dist+=temp_h*temp_h;
                count++;
            }
        }
    return dist/count;
}

void idm_dist(float *img1, float *img2, float *out, 
        int im_row, int im_col, int patch_row, int patch_col,
        int warp_row, int warp_col){
    /* assume same size img */
    float best_dis;
    float temp;
    int x,y,xx,yy,B_x,B_y;
    
    /* 3) Return distance */
    float* dis = mxGetPr(plhs[0]);
    int count=0;
    for (y=0; y<im_col; y++){
        for (x=0; x<im_row; x++)        
            {
                best_dis=FLT_MAX;
                for(xx=x-warp_row; xx<=x+warp_row; xx++){
                    for(yy=y-warp_col; yy<=y+warp_col; yy++){
                        if(xx >= 0 && yy >= 0 && xx < im_row && yy < im_col){
                            temp=pixel_distance(x, y, xx, yy, im_row, im_col, patch_row, patch_col, img1, img2);
                            if(temp<best_dis)
                            {
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
