% clear all;

load('preprocessed data\HRTF_data.mat','HRTF_data');

low_dim=10;
[no_of_samples,length_training_set,no_of_directions,no_of_channels]=size(HRTF_data);

PC_mtx=zeros(no_of_samples,low_dim,no_of_directions,no_of_channels);
weight_vectors = zeros(low_dim,length_training_set,no_of_directions,no_of_channels);
  

disp('Calulating eigen vectors')
for m=1:no_of_channels
    channel=m;
    for n=1:no_of_directions
    direction=n;


data_mtx= HRTF_data(:,:,direction,channel);
cov_mtx = data_mtx'*data_mtx;
%calculating sigular vectors and corresponding sigular values
[eigen_vector,eigen_value_mtx] = eig(cov_mtx);
[eigen_value,sorted_index]=sort(diag(eigen_value_mtx),'descend');
temp=eigen_vector;

for i=1:length(sorted_index)
  eigen_vector(:,i)=temp(:,sorted_index(i,1));   
end
 
%eigen vectors
eigen_vector= data_mtx*eigen_vector;
for i=1:length_training_set
 eigen_vector(:,i)= eigen_vector(:,i)/norm(eigen_vector(:,i));  %normalized eigenvectors
end

eigen_value =(eigen_value/sum(eigen_value))*100;%normalized eigen values


 weight_vectors(:,:,n,m) = eigen_vector(:,1:low_dim)'*data_mtx;%low dimention projection
 PC_mtx(:,:,n,m) =  eigen_vector(:,1:low_dim);   
    
    end
end

disp('SAVING DATA');
save('preprocessed data\weight_vectors.mat','weight_vectors');
save('preprocessed data\PC_mtx.mat','PC_mtx');
disp('DATA SAVED');

clear all;