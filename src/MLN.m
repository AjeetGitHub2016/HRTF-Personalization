


[length_input,dim2,length_channels]=size(ANTHRO_data);

[dim1,length_samples]=size(weight_vectors(:,:,direction,channel));
length_output=dim1+1;

length_hidden=20;

length_epoch=3000;
learning_rate=.08;


output=zeros(length_output,length_samples);

sqerror=zeros(length_epoch,1);

weights_input_stage=rand(length_input,length_hidden)-0.5;
weights_output_stage=rand(length_hidden,length_output)-0.5;

samples_input= ANTHRO_data(:,1:length_samples,1);
samples_output=[weight_vectors(:,:,direction,channel);ITD_data(direction,:)];

output_error=zeros(length_output,1);
input_error=zeros(length_hidden,1);

disp('Training Multilayer Neural Network');

for i=1:length_output
samples_output(i,:)=samples_output(i,:)/norm(samples_output(i,:));
end

for loop=1:length_epoch
    for l=1:length_samples
    
    input=samples_input(:,l);
    desired_output=samples_output(:,l);
   
    hidden_input=weights_input_stage'*input;
    hidden_output=scaling*hidden_input./sqrt(1+hidden_input.^2);
    
    last_input=weights_output_stage'*hidden_output;
    output(:,l)=scaling*last_input./sqrt(1+last_input.^2);
   
%weight updation error calculation

for i=1:length_output
   output_error(i,1)=scaling*(desired_output(i)-output(i,l))/((1+last_input(i)^2)^1.5);
end

weights_output_stage=weights_output_stage+learning_rate*hidden_output*output_error';

for i=1:length_hidden
temp=0;   
for j=1:length_output
    temp=temp+output_error(j)*weights_output_stage(i,j);
end
input_error(i,1)=scaling*temp/((1+hidden_input(i)^2)^1.5);
end


  weights_input_stage=weights_input_stage+learning_rate*input*input_error';
 

for i=1:length_output
   sqerror(loop)=sqerror(loop)+(desired_output(i)-output(i,l))^2; 
end
sqerror(loop)=sqerror(loop)/2;

    end
end

disp('Training Completed');

for i=1:length_output
%output(i,:)=output(i,:)/norm(output(i,:));
end

%{
x=1:1:length_output-1;
figure;plot(x,samples_output(1:length_output-1,2),x,output(1:length_output-1,2));
legend('desired output','output of training data set');


x=1:1:length_epoch;
figure;plot(x,sqerror);
legend('mean square error');
%}

%MLN_OUTPUT;
  

 