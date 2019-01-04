#include<iostream>
#include<fstream>
#include<string>
#include<algorithm>
#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<limits>
#include<iomanip>

using namespace std;

#define BLOCK_SIZE 8


__global__ void kernel(int *dim,float *k_k, int *Xlocation, int *Ylocation, int *Zlocation, int *Width, int *Height, int *Depth, float *FixedTemperature, float *OldTemp, float *NewTemp, int *Xaxis, int *Yaxis, int *Zaxis, int *HeatSources){



    if (*dim ==50){

        __shared__ float temp[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
        int j = blockIdx.x*blockDim.x + threadIdx.x;
        int i = blockIdx.y*blockDim.y + threadIdx.y;
        int leni, lenj;

        if (i <= *Yaxis && j <= *Xaxis){

            int threadj = threadIdx.x + 1;
            int threadi = threadIdx.y + 1;

            temp[threadi][threadj] = OldTemp[j+i*(*Yaxis)];

            if (blockIdx.x==(int)*Xaxis/BLOCK_SIZE){

                lenj = *Xaxis%BLOCK_SIZE;


            }
            else{

                lenj = BLOCK_SIZE;
            }

            if (blockIdx.y ==(int)*Yaxis/BLOCK_SIZE){

                leni = *Yaxis%BLOCK_SIZE;
            }
            else{

                leni = BLOCK_SIZE;
            }



            if (threadIdx.x < 1){

                if (j < 1){

                    temp[threadi][threadj - 1] = *(OldTemp + i*(*Yaxis) + j);
                }
                else{
                    temp[threadi][threadj - 1] = *(OldTemp + i*(*Yaxis) + j - 1);
                }

                if (j >= *Xaxis - lenj){
                    temp[threadi][threadj + lenj] = *(OldTemp + i*(*Yaxis)+j +lenj-1);


                }
                else{
                    temp[threadi][threadj + lenj] = *(OldTemp + i*(*Yaxis)+j +lenj);
                }


            }


            if (threadIdx.y < 1 ){

                if(i<1){

                    temp[threadi - 1][threadj] = *(OldTemp + i*(*Yaxis)+j);

                }
                else{

                    temp[threadi - 1][threadj] = *(OldTemp + (i-1)*(*Yaxis)+j);

                }

                if(i >= (*Yaxis) - leni){

                    temp[threadi+leni][threadj] = *(OldTemp+(i+leni-1)*(*Yaxis)+j);


                }
                else{

                    temp[threadi+leni][threadj] = *(OldTemp+(i+leni)*(*Yaxis)+j);
                }

            }

            __syncthreads();

        if (i < *Yaxis && j < *Xaxis){

            NewTemp[j+i*(*Yaxis)] = temp[threadi][threadj] + *k_k*(temp[threadi-1][threadj] + temp[threadi][threadj -1] + temp[threadi+1][threadj] + temp[threadi][threadj+1] - 4*temp[threadi][threadj]);



            for (int p = 0; p < *HeatSources; p++){

                if((i > Ylocation[p]-1) && (i <= Ylocation[p] + Height[p]-1) && (j > Xlocation[p]-1) && (j <= Xlocation[p] + Width[p]-1)){

                    NewTemp[j+(i)*(*Yaxis)] = FixedTemperature[p];

                }


            }

            OldTemp[j+i*(*Yaxis)] = NewTemp[j+i*(*Yaxis)];


        }

        }
    }

    else if (*dim == 51){

        __shared__ float temp[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2];
        int j = blockIdx.x*blockDim.x + threadIdx.x;
        int i = blockIdx.y*blockDim.y + threadIdx.y;
        int k = blockIdx.z*blockDim.z + threadIdx.z;
        int leni, lenj, lenk;

        if (i <= *Yaxis && j <= *Xaxis && k <= *Zaxis){


            int threadj = threadIdx.x + 1;
            int threadi = threadIdx.y + 1;
            int threadk = threadIdx.z + 1;

            temp[threadi][threadj][threadk] = OldTemp[(j+i*(*Yaxis))*(*Zaxis) + k];

            if (blockIdx.x==(int)*Xaxis/BLOCK_SIZE){

                lenj = *Xaxis%BLOCK_SIZE;


            }
            else{

                lenj = BLOCK_SIZE;
            }

            if (blockIdx.y ==(int)*Yaxis/BLOCK_SIZE){

                leni = *Yaxis%BLOCK_SIZE;
            }
            else{

                leni = BLOCK_SIZE;
            }

            if (blockIdx.z == (int)*Zaxis/BLOCK_SIZE){
                lenk = *Zaxis%BLOCK_SIZE;
            }
            else{
                lenk = BLOCK_SIZE;
            }



            if (threadIdx.x < 1){

                if (j < 1){

                    temp[threadi][threadj - 1][threadk] = *(OldTemp + (j+i*(*Yaxis))*(*Zaxis) + k);
                }
                else{
                    temp[threadi][threadj - 1][threadk] = *(OldTemp + (j-1+i*(*Yaxis))*(*Zaxis) + k);
                }

                if (j >= *Xaxis - lenj){
                    temp[threadi][threadj + lenj][threadk] = *(OldTemp + (j+lenj - 1 + i*(*Yaxis))*(*Zaxis) + k);


                }
                else{
                    temp[threadi][threadj + lenj][threadk] = *(OldTemp + (j+lenj + i*(*Yaxis))*(*Zaxis) + k);
                }


            }



            if (threadIdx.y < 1 ){

                if(i<1){

                    temp[threadi - 1][threadj][threadk] = *(OldTemp + (j+i*(*Yaxis))*(*Zaxis) + k);

                }
                else{

                    temp[threadi - 1][threadj][threadk] = *(OldTemp + (j+(i-1)*(*Yaxis))*(*Zaxis) + k);

                }

                if(i >= (*Yaxis) - leni){

                    temp[threadi+leni][threadj][threadk] = *(OldTemp + (j+(i + leni-1)*(*Yaxis))*(*Zaxis) + k);


                }
                else{

                    temp[threadi+leni][threadj][threadk] = *(OldTemp + (j+(i + leni)*(*Yaxis))*(*Zaxis) + k);
                }

            }

            if (threadIdx.z < 1 ){

                if(k<1){

                    temp[threadi][threadj][threadk-1] = *(OldTemp + (j+i*(*Yaxis))*(*Zaxis) + k);
                }
                else{

                    temp[threadi][threadj][threadk-1] = *(OldTemp + (j+i*(*Yaxis))*(*Zaxis) + k - 1);

                }

                if(k >= (*Zaxis) - lenk){

                    temp[threadi][threadj][threadk + lenk] = *(OldTemp + (j+i*(*Yaxis))*(*Zaxis) + k + lenk - 1);


                }
                else{

                    temp[threadi][threadj][threadk + lenk] = *(OldTemp + (j+i*(*Yaxis))*(*Zaxis) + k + lenk);
                }

            }

            __syncthreads();

            if (i < *Yaxis && j < *Xaxis && k<*Zaxis){

                NewTemp[(j+i*(*Yaxis))*(*Zaxis) + k] = temp[threadi][threadj][threadk] + *k_k*(temp[threadi-1][threadj][threadk] + temp[threadi][threadj -1][threadk] + temp[threadi+1][threadj][threadk] + temp[threadi][threadj+1][threadk]+temp[threadi][threadj][threadk-1]+temp[threadi][threadj][threadk+1] - 6*temp[threadi][threadj][threadk]);

                for (int p = 0; p < *HeatSources; p++){


                    if((i > Ylocation[p]-1) && (i <= Ylocation[p] + Height[p]-1) && (j > Xlocation[p]-1) && (j <= Xlocation[p] + Width[p]-1)&& (k >= Zlocation[p])&& (k < Zlocation[p] + Depth[p])){

                        NewTemp[(j+i*(*Yaxis))*(*Zaxis) + k] = FixedTemperature[p];

                    }

                }

                OldTemp[(j+i*(*Yaxis))*(*Zaxis) + k] = NewTemp[(j+i*(*Yaxis))*(*Zaxis) + k];
            }
        }

    }

}

int main(int argc, char const *argv[])
{
    int StringLength = 0;

    ifstream inFile(argv[1]);
    string strOneLine;
    int ParaNumber= 0;
    int dim = 0;
    vector <string> parastring;
    float k_k, StartTemp, FixedTemperature[26];
    int TimeSteps, Xaxis, Yaxis, Zaxis, Xlocation[26], Ylocation[26], Zlocation[26], Width[26], Height[26], Depth[26];
    int HeatSources;
    string CurrString;
    int len;

    while(inFile)
    {

        getline(inFile, strOneLine);
        StringLength = strOneLine.length();
        if (StringLength==0|| int(strOneLine[0]) == 13) continue;



        for(int i = 0; i<StringLength; i ++){

            if (strOneLine.at(i) == ' ') continue;

            if (strOneLine.at(i) == '#')
            {
                break;
            }

            if (strOneLine.at(i) == ',') {

                ParaNumber = ParaNumber + 1;
                parastring.push_back(CurrString);
                CurrString.clear();
                continue;
            }

            if (ParaNumber == 0){

                dim = int(strOneLine.at(i));
                ParaNumber=ParaNumber+1;
                break;

            }


            else
            {

                CurrString.push_back(strOneLine.at(i));
           }

        }
        if(CurrString.length()!= 0){

            parastring.push_back(CurrString);
            CurrString.clear();
            ParaNumber = ParaNumber + 1;

        }


    }

    int *d_Xlocation, *d_Ylocation, *d_Zlocation, *d_Width, *d_Height, *d_Depth, *d_Xaxis, *d_Yaxis, *d_Zaxis, *d_dim, *d_HeatSources;
    float *d_OldTemp, *d_NewTemp, *d_k, *d_FixedTemperature;

    if (dim==50)
    {

        parastring[0].insert(0,1,'0');
        k_k = atof(parastring[0].c_str());
        TimeSteps = atoi(parastring[1].c_str());
        Xaxis = atoi(parastring[2].c_str());
        Yaxis = atoi(parastring[3].c_str());
        StartTemp = atof(parastring[4].c_str());
        HeatSources = (ParaNumber-5)/5;

        for (int i=5;i<ParaNumber-5;i=i+5){

            Xlocation[(i-5)/5] = atoi(parastring[i].c_str());
            Ylocation[(i-5)/5] = atoi(parastring[i+1].c_str());
            Width[(i-5)/5] = atoi(parastring[i+2].c_str());
            Height[(i-5)/5] = atoi(parastring[i+3].c_str());
            FixedTemperature[(i-5)/5] = atof(parastring[i+4].c_str());


        }
        //cout<<"Dimension = "<<dim<<endl;
        //cout<<"k = "<<k_k<<endl;
        //cout<<"Time Steps = "<<TimeSteps<<endl;
        //cout<<"X Axis, Y axis = "<<Xaxis<<", "<<Yaxis<<endl;
        //cout<<"Starting Temp = "<<StartTemp<<endl;

        //for (int i=0; i<HeatSources; i++)
        //{
        //    cout<<Xlocation[i]<<", "<<Ylocation[i]<<", "<<Width[i]<<", "<<Height[i]<<", "<<FixedTemperature[i]<<endl;
        //}

        if (Xaxis >= Yaxis){

            len = Xaxis*Xaxis;
        }
        else{
            len = Yaxis*Yaxis;
        }

        int size = (len)*sizeof(float);

        float OldTemp[len] = {0};
        float NewTemp[len] = {0};

        for (int i = 0; i < Yaxis ; i ++){
            for (int j = 0; j < Xaxis; j ++){
                OldTemp[i*Yaxis + j] = StartTemp;

                for (int p = 0; p < HeatSources; p++){

                    if((i > Ylocation[p]-1) && (i <= Ylocation[p] + Height[p]-1) && (j > Xlocation[p]-1) && (j <= Xlocation[p] + Width[p]-1)){

                        OldTemp[i*Yaxis + j] = FixedTemperature[p];

                    }
                }
            }
        }

        cudaMalloc((void **)&d_OldTemp, size);
        cudaMalloc((void **)&d_NewTemp, size);
        cudaMalloc((void **)&d_Xlocation, HeatSources*sizeof(int));
        cudaMalloc((void **)&d_Ylocation, HeatSources*sizeof(int));
        cudaMalloc((void **)&d_Xaxis, sizeof(int));
        cudaMalloc((void **)&d_Yaxis, sizeof(int));
        cudaMalloc((void **)&d_k, sizeof(float));
        cudaMalloc((void **)&d_Width, HeatSources*sizeof(int));
        cudaMalloc((void **)&d_Height, HeatSources*sizeof(int));
        cudaMalloc((void **)&d_FixedTemperature, HeatSources*sizeof(float));
        cudaMalloc((void **)&d_Zlocation, HeatSources*sizeof(int));
        cudaMalloc((void **)&d_Depth, HeatSources*sizeof(int));
        cudaMalloc((void **)&d_Zaxis, sizeof(int));
        cudaMalloc((void **)&d_dim, sizeof(int));
        cudaMalloc((void **)&d_HeatSources, sizeof(int));

        cudaMemcpy(d_OldTemp, OldTemp, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_NewTemp, NewTemp, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Xlocation, Xlocation, HeatSources*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Ylocation, Ylocation, HeatSources*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Xaxis, &Xaxis, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Yaxis, &Yaxis, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, &k_k, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Width, Width, HeatSources*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Height, Height, HeatSources*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_FixedTemperature, FixedTemperature, HeatSources*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Zlocation, Zlocation, HeatSources*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Zaxis, &Zaxis, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Depth, Depth, HeatSources*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dim, &dim, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_HeatSources, &HeatSources, sizeof(int), cudaMemcpyHostToDevice);

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((int)ceil(((Xaxis + BLOCK_SIZE - 1)/BLOCK_SIZE)),(int)(ceil(((Yaxis + BLOCK_SIZE - 1)/BLOCK_SIZE))));

        for (int i = 0; i < TimeSteps; i++){

            kernel<<<grid,block>>>(d_dim, d_k, d_Xlocation, d_Ylocation, d_Zlocation, d_Width, d_Height, d_Depth, d_FixedTemperature, d_OldTemp, d_NewTemp, d_Xaxis, d_Yaxis, d_Zaxis, d_HeatSources);

        }
        cudaMemcpy(NewTemp, d_NewTemp, size, cudaMemcpyDeviceToHost);

        cudaFree(d_OldTemp);
        cudaFree(d_NewTemp);
        cudaFree(d_Xlocation);
        cudaFree(d_Ylocation);
        cudaFree(d_Xaxis);
        cudaFree(d_Yaxis);
        cudaFree(d_k);
        cudaFree(d_Width);
        cudaFree(d_Height);
        cudaFree(d_FixedTemperature);
        cudaFree(d_Zlocation);
        cudaFree(d_Zaxis);
        cudaFree(d_Depth);
        cudaFree(d_dim);
        cudaFree(d_HeatSources);


        ofstream build ("heatOutput.csv", std::ofstream::out);
        for(int i = 0; i <Yaxis; i++){
            for(int j = 0; j <Xaxis - 1; j++){

                build<<NewTemp[i*Yaxis + j]<<", ";

            }
            build<<NewTemp[i*Yaxis + Xaxis-1];
            if(i != Yaxis -1){

                build<<endl;

            }
        }
        build.close();

    }

    if (dim ==51)
    {
        parastring[0].insert(0,1,'0');
        k_k = atof(parastring[0].c_str());
        TimeSteps = atoi(parastring[1].c_str());
        Xaxis = atoi(parastring[2].c_str());
        Yaxis = atoi(parastring[3].c_str());
        Zaxis = atoi(parastring[4].c_str());
        StartTemp = atof(parastring[5].c_str());
        HeatSources = (ParaNumber-7)/5;


        for (int i=6;i<ParaNumber-6;i=i+7){

            Xlocation[(i-6)/7] = atoi(parastring[i].c_str());
            Ylocation[(i-6)/7] = atoi(parastring[i+1].c_str());
            Zlocation[(i-6)/7] = atoi(parastring[i+2].c_str());
            Width[(i-6)/7] = atoi(parastring[i+3].c_str());
            Height[(i-6)/7] = atoi(parastring[i+4].c_str());
            Depth[(i-6)/7] = atoi(parastring[i+5].c_str());
            FixedTemperature[(i-6)/7] = atof(parastring[i+6].c_str());

        }

        //cout<<"Dimension = "<<dim<<endl;
        //cout<<"k = "<<k_k<<endl;
        //cout<<"Time Steps = "<<TimeSteps<<endl;
        //cout<<"X Axis, Y axis, Z axis = "<<Xaxis<<", "<<Yaxis<<", "<<Zaxis<<endl;
        //cout<<"Starting Temp = "<<StartTemp<<endl;
        //cout<<"HeatSources = "<<HeatSources<<endl;

        //for (int i=0; i<HeatSources; i++)
        //{
          //  cout<<Xlocation[i]<<", "<<Ylocation[i]<<", "<<Zlocation[i]<<", "<<Width[i]<<", "<<Height[i]<<", "<<Depth[i]<<", "<<FixedTemperature[i]<<endl;
        //}

        if (Xaxis >= Yaxis && Xaxis >= Zaxis){

            len = Xaxis*Xaxis*Xaxis;
        }
        else if (Yaxis >= Xaxis && Yaxis >= Zaxis){
            len = Yaxis*Yaxis*Yaxis;
        }
        else if (Zaxis>= Yaxis &&  Zaxis>=Yaxis){

            len = Zaxis*Zaxis*Zaxis;
        }

        int size = (len)*sizeof(float);

        float OldTemp[len] = {0};
        float NewTemp[len] = {0};

        for(int k = 0; k <Zaxis; k++){

            for (int i = 0; i < Yaxis ; i ++){
                for (int j = 0; j < Xaxis; j ++){
                    OldTemp[(j+i*(Yaxis))*(Zaxis) + k] = StartTemp;

                    for (int p = 0; p < HeatSources; p++){

                        if((i > Ylocation[p]-1) && (i <= Ylocation[p] + Height[p]-1) && (j > Xlocation[p]-1) && (j <= Xlocation[p] + Width[p]-1) && (k > Zlocation[p] - 1) && (k<= Zlocation[p] + Depth[p] - 1)){

                            OldTemp[(j+i*(Yaxis))*(Zaxis) + k] = FixedTemperature[p];

                        }
                    }
                }
            }
        }

        cudaMalloc((void **)&d_OldTemp, size);
        cudaMalloc((void **)&d_NewTemp, size);
        cudaMalloc((void **)&d_Xlocation, HeatSources*sizeof(int));
        cudaMalloc((void **)&d_Ylocation, HeatSources*sizeof(int));
        cudaMalloc((void **)&d_Xaxis, sizeof(int));
        cudaMalloc((void **)&d_Yaxis, sizeof(int));
        cudaMalloc((void **)&d_k, sizeof(float));
        cudaMalloc((void **)&d_Width, HeatSources*sizeof(int));
        cudaMalloc((void **)&d_Height, HeatSources*sizeof(int));
        cudaMalloc((void **)&d_FixedTemperature, HeatSources*sizeof(float));
        cudaMalloc((void **)&d_Zlocation, HeatSources*sizeof(int));
        cudaMalloc((void **)&d_Depth, HeatSources*sizeof(int));
        cudaMalloc((void **)&d_Zaxis, sizeof(int));
        cudaMalloc((void **)&d_dim, sizeof(int));
        cudaMalloc((void **)&d_HeatSources, sizeof(int));

        cudaMemcpy(d_OldTemp, OldTemp, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_NewTemp, NewTemp, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Xlocation, Xlocation, HeatSources*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Ylocation, Ylocation, HeatSources*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Xaxis, &Xaxis, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Yaxis, &Yaxis, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, &k_k, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Width, Width, HeatSources*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Height, Height, HeatSources*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_FixedTemperature, FixedTemperature, HeatSources*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Zlocation, Zlocation, HeatSources*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Zaxis, &Zaxis, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Depth, Depth, HeatSources*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dim, &dim, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_HeatSources, &HeatSources, sizeof(int), cudaMemcpyHostToDevice);

        dim3 block(BLOCK_SIZE, BLOCK_SIZE,BLOCK_SIZE);
        dim3 grid((int)ceil(((Xaxis + BLOCK_SIZE - 1)/BLOCK_SIZE)),(int)(ceil(((Yaxis + BLOCK_SIZE - 1)/BLOCK_SIZE))),(int)(ceil(((Zaxis + BLOCK_SIZE - 1)/BLOCK_SIZE))));
        for (int i = 0; i < TimeSteps; i++){

            kernel<<<grid,block>>>(d_dim, d_k, d_Xlocation, d_Ylocation, d_Zlocation, d_Width, d_Height, d_Depth, d_FixedTemperature, d_OldTemp, d_NewTemp, d_Xaxis, d_Yaxis, d_Zaxis, d_HeatSources);

        }
        cudaMemcpy(NewTemp, d_NewTemp, size, cudaMemcpyDeviceToHost);

        cudaFree(d_OldTemp);
        cudaFree(d_NewTemp);
        cudaFree(d_Xlocation);
        cudaFree(d_Ylocation);
        cudaFree(d_Xaxis);
        cudaFree(d_Yaxis);
        cudaFree(d_k);
        cudaFree(d_Width);
        cudaFree(d_Height);
        cudaFree(d_FixedTemperature);
        cudaFree(d_Zlocation);
        cudaFree(d_Zaxis);
        cudaFree(d_Depth);
        cudaFree(d_dim);
        cudaFree(d_HeatSources);


        ofstream build ("heatOutput.csv", std::ofstream::out);
        for (int l = 0; l < Zaxis; l++){
            for(int i = 0; i <Yaxis; i++){
                for(int j = 0; j <Xaxis - 1; j++){

                    build<<NewTemp[(j+i*(Yaxis))*(Zaxis) + l]<<", ";

                }
                build<<NewTemp[(Xaxis - 1 +i*(Yaxis))*(Zaxis) + l];

                if(l != Zaxis -1 || i != Yaxis -1){

                    build<<endl;
                }
            }

            if (l != Zaxis - 1){
                build<<endl;

            }

        }
        build.close();

    }

    return 0;
}






