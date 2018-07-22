clc;
clear all;
close all;

disp('loading HRIR data and preparing input training set for ANN');
load 'data set\anthropometry\anthro.mat'

length_training_set=30;
no_of_subjects=37;
ANTHRO_data=zeros(8,no_of_subjects,2);

for i=1:no_of_subjects
ANTHRO_data(1,i,:)=X(i,1);%anthroparametric data for head related parameters
ANTHRO_data(2,i,:)=X(i,3);
ANTHRO_data(3,i,:)=X(i,12);
%anthroparametric data for pinnae related parameters
%left ear 

ANTHRO_data(4,i,1)=D(i,1);
ANTHRO_data(5,i,1)=D(i,3);
ANTHRO_data(6,i,1)=D(i,4);
ANTHRO_data(7,i,1)=D(i,5);
ANTHRO_data(8,i,1)=D(i,6);

%right ear
ANTHRO_data(4,i,2)=D(i,1+8);
ANTHRO_data(5,i,2)=D(i,3+8);
ANTHRO_data(6,i,2)=D(i,4+8);
ANTHRO_data(7,i,2)=D(i,5+8);
ANTHRO_data(8,i,2)=D(i,6+8);
end


disp('prepairing input training set done');
disp('loading HRIR data');

no_of_channels=2;% for left and right ear
no_of_samples=200;%time instant
no_of_directions=25*50;%toatal directions


HRIR_data=zeros(no_of_samples,length_training_set,no_of_directions,no_of_channels);

ITD_data=zeros(no_of_directions,length_training_set); 

for i=1:length_training_set

    file_name=['data set\hrir\subject_' num2str(i) '\hrir_final.mat'];
       load(file_name);
    %creating ITD data matrix for 30 subjects 
   [dim1,dim2]=size(ITD);
    for j=1:dim1
      ITD_data((dim2*(j-1)+1):dim2*j,i)=ITD(j,:)';
    end
     %creating 4-dimentional HRIR_data matrix (200*30*1250*2)  
    for j=1:2
        if(j==1)
            [dim1,dim2,dim3]= size(hrir_l);
            temp=hrir_l;
        else
            [dim1,dim2,dim3]= size(hrir_r);
            temp=hrir_r;
        end
        
        direction=1;
        for k=1:dim1
            for l=1:dim2
                HRIR_data(:,i,direction,j)=temp(k,l,:);
                direction=direction+1;
            end
        end
    end 
end

disp('loading Done');
disp('Computing HRTF log10');

direction=direction-1;
HRTF_data=zeros(no_of_samples,length_training_set,no_of_directions,no_of_channels);
%taking log magnitude of HRTF for 30 subjects at all directions
for i=1:2
   for j=1:direction
       for k=1:length_training_set
        HRTF_data(:,k,j,i)=log10(abs(fft(HRIR_data(:,k,j,i))));
       end
   end
end
 
disp('Computing HRTF log10 Done');
disp('Computing HRTF log10 mean');
%calculating directional mean
mean_vector=zeros(no_of_samples,length_training_set,no_of_channels);
for i=1:2
   for j=1:length_training_set
       temp=zeros(no_of_samples,1);
       for k=1:direction
         temp=temp+HRTF_data(:,j,k,i);
       end
       temp=temp/direction;
       mean_vector(:,j,i)=temp;
   end
end

disp('Computing HRTF log10 mean done');
disp('Subtracting mean');
%calculating log directional HRTF
for i=1:2
   for j=1:length_training_set
       for k=1:direction
          HRTF_data(:,j,k,i)=HRTF_data(:,j,k,i)-mean_vector(:,j,i);
       end
   end
end

disp('Subtraction Done');
disp('Computing HRTF log10 dir mean');
% calculating mean for all subjects
directional_mean_vector=zeros(no_of_samples,direction,no_of_channels);
for i=1:2
   for j=1:direction
       temp=zeros(no_of_samples,1);
       for k=1:length_training_set
         temp=temp+HRTF_data(:,k,j,i);
       end
       temp=temp/length_training_set;
       directional_mean_vector(:,j,i)=temp;
   end
end

disp('Subtracting dir mean');
disp('Computing HRTF log10 dir mean Done');
% calculating PTF
for i=1:2
   for j=1:direction
       for k=1:length_training_set
       HRTF_data(:,k,j,i)=HRTF_data(:,k,j,i)-directional_mean_vector(:,j,i);
       end
   end
end

disp('Subtracting dir mean done');

disp('SAVING PREPROCESSED DATA');
%saving data matrixs
save('preprocessed data\ANTHRO_data.mat','ANTHRO_data');
save('preprocessed data\ITD_data.mat','ITD_data');
save('preprocessed data\HRIR_data.mat','HRIR_data');
save('preprocessed data\HRTF_data.mat','HRTF_data');
save('preprocessed data\mean_vector.mat','mean_vector');
save('preprocessed data\directional_mean_vector.mat','directional_mean_vector');
disp('DATA SAVED');

clear all;
disp('DATA PREPROCESSING DONE');