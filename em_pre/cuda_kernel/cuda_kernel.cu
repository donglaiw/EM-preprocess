////
//// Created by matinraayai on 7/18/18.
////
//#define BLOCK_X  16
//#define BLOCK_Y  16
//
//#define s2(a,b)            { float tmp = a; a = min(a,b); b = max(tmp,b); }
////////////////////////////////////////////////////////////////////////////////
///**
// *
// * @param d_out
// * @param d_in
// * @param NX
// * @param NY
// * @return
// */
//__global__ void medianFilter(
//        float *       d_out,
//        const float * d_in,
//        const int     NX,       // Number of rows
//        const int     NY)       // Number of cols
//{
// const int tx = threadIdx.x;
// const int ty = threadIdx.y;
//
// // Guards, at the boundary?
// bool is_x_top = (tx == 0);
// bool is_x_bot = (tx == BLOCK_X-1);
// bool is_y_top = (ty == 0);
// bool is_y_bot = (ty == BLOCK_Y-1);
//
// if (is_x_top)       SMEM(tx-1, ty-1) = 0;
// if (is_y_top) {         SMEM(tx  , ty-1) = 0;
// else if (is_x_bot)  SMEM(tx+1, ty-1) = 0;
// if (is_x_top)       SMEM(tx-1, ty+1) = 0;
// } else if (is_y_bot) {  SMEM(tx  , ty+1) = 0;
//  __shared__ float smem[BLOCK_X+2][BLOCK_Y+2];
//  else if (is_x_bot)  SMEM(tx+1, ty+1) = 0;
//  }
// if (is_x_top)           SMEM(tx-1, ty  ) = 0;
// else if (is_x_bot)      SMEM(tx+1, ty  ) = 0;
//
//      // x,y are 1 based indicies, the macros IN, OUT subtract 1
// // Clear out shared memory (zero padding)int x = blockIdx.x * blockDim.x + tx;int y = blockIdx.y * blockDim.y + ty;// Guards, at the boundary and still more image to process?is_x_top &= (x > 0);is_x_bot &= (x < NX);is_y_bot &= (y < NY);is_y_top &= (y > 0);// Each thread reads the input matrix.if (is_y_top) {         SMEM(tx  , ty-1) = IN(x  , y-1);         else if (is_x_bot)  SMEM(tx+1, ty-1) = IN(x+1, y-1);         if (is_x_top)       SMEM(tx-1, ty+1) = IN(x-1, y+1);     } else if (is_y_bot) {  SMEM(tx  , ty+1) = IN(x  , y+1);         else if (is_x_bot)  SMEM(tx+1, ty+1) = IN(x+1, y+1);     }
//         SMEM(tx  , ty  ) = IN(x  , y  ); // selfif (is_x_top)           SMEM(tx-1, ty  ) = IN(x-1, y  );if (is_x_top)       SMEM(tx-1, ty-1) = IN(x-1, y-1);else if (is_x_bot)      SMEM(tx+1, ty  ) = IN(x+1, y  );__syncthreads();// Pull top six values from shared memoryfloat v[6] ={         SMEM(tx  , ty  ),    //     sel4         SMEM(tx+1, ty  )     //  S         SMEM(tx-1, ty  ),    //  9         SMEM(tx-1, ty-1),    //  NW     (North West neighbor0         SMEM(tx  , ty-1),    //   1         SMEM(tx+1, ty-1),    //  S5     };mnmx6(v[0], v[1], v[2], v[3], v[4], v[5]);// With each pass, remove min and max values and add new value
//      // Replace Max with new value.
//
//      v[5] = SMEM(tx-1, ty+1);    // NE
//
//      mnmx5(v[1], v[2], v[3], v[4], v[5]);
//
//      v[5] = SMEM(tx  , ty+1);    //  E
//
//      mnmx4(v[2], v[3], v[4], v[5]);
//
//      v[5] = SMEM(tx+1, ty+1);    // SE
//
//      mnmx3(v[3], v[4], v[5]);
//      if(x >= 1 && x <= NX && y >= 1 && y <= NY) { 8
//  {
//  }
//      // v[4] now contains the middle value.
//       // Guard against indicies out of range.
//           OUT(x,y) = v[4];
//      }
//
//  }
//#define mn3(a,b,c)         s2(a,b); s2(a,c);
//
//#define mx3(a,b,c)         s2(b,c); s2(a,c);
//#define mnmx3(a,b,c)       mx3(a,b,c); s2(a,b);                               // 3 exchanges
//#define mnmx5(a,b,c,d,e)   s2(a,b); s2(c,d); mn3(a,c,e); mx3(b,d,e);          // 6 exchanges
//#define mnmx4(a,b,c,d)     s2(a,b); s2(c,d); s2(a,c); s2(b,d);                // 4 exchanges
//
//#define mnmx6(a,b,c,d,e,f) s2(a,d); s2(b,e); s2(c,f); mn3(a,b,c); mx3(d,e,f); // 7 exchanges
//
//#define SMEM(x,y)  smem[(x)+1][(y)+1]
//#define  IN(x,y)    d_in[((y)-1) + ((x)-1) * NY]
//
//
//do someting
//#define OUT(x,y)   d_out[((y)-1) + ((x)-1) * NY]
//
//
//
//        __global__ int foo(int* a, int* b. int*c ) {
//}
//
//
//
//