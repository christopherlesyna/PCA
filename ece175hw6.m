clear all
warning('off')
%%{
data = load('data.mat');
label = load('label.mat');

imageTrain = data.imageTrain;
imageTest = data.imageTest;
labelTest = label.labelTest;
labelTrain = label.labelTrain;
%}

%Part 1 (RIGHT)
%%{
load data.mat
load label.mat
m=zeros(784,1);
for i=1:5000
    train=imageTrain(:,:,i);
    m=m+train(:);
end
m=m(:)/(5000);
scov=zeros(784,784,1);
for i=1:5000
    train=imageTrain(:,:,i);
    d=train(:) - m;
    scov=scov + d * d';
end
scov = scov/5000;

[scov_eigvec scov_eigval] = eig(scov);
%scov_eigval = scov_eigval.^2;
maxval = max(scov_eigval(:));

component1=reshape(scov_eigvec(:,784),[28 28]);
component2=reshape(scov_eigvec(:,783),[28 28]);
component3=reshape(scov_eigvec(:,782),[28 28]);
component4=reshape(scov_eigvec(:,781),[28 28]);
component5=reshape(scov_eigvec(:,780),[28 28]);
component6=reshape(scov_eigvec(:,779),[28 28]);
component7=reshape(scov_eigvec(:,778),[28 28]);
component8=reshape(scov_eigvec(:,777),[28 28]);
component9=reshape(scov_eigvec(:,776),[28 28]);
component10=reshape(scov_eigvec(:,775),[28 28]);
figure(1)
subplot(2,5,1);
imshow(component1,[])
subplot(2,5,2);
imshow(component2,[])
subplot(2,5,3);
imshow(component3,[])
title('Principal Components')
subplot(2,5,4);
imshow(component4,[])
subplot(2,5,5);
imshow(component5,[])
subplot(2,5,6);
imshow(component6,[])
subplot(2,5,7);
imshow(component7,[])
subplot(2,5,8);
imshow(component8,[])
subplot(2,5,9);
imshow(component9,[])
subplot(2,5,10);
imshow(component10,[])

scov_eigval_vector = zeros(784,1);
for vec1=1:784
    scov_eigval_vector(vec1) = scov_eigval(785-vec1,785-vec1);
    vec1=vec1+1;
end

figure(2)
scatter([1:784],[scov_eigval_vector])
title('Eigenvalues In Decreasing Order')

mdig5=zeros(784,1);
for i=1:5000
    if labelTrain(i) == 5
    traindig5=imageTrain(:,:,i);
    mdig5=mdig5+traindig5(:);
    end
end
mdig5=mdig5(:)/(sum(labelTrain(:)==5));
scovdig5=zeros(784,784,1);
for i=1:5000
    if labelTrain(i)==5
    traindig5=imageTrain(:,:,i);
    d5=traindig5(:) - mdig5;
    scovdig5=scovdig5 + d5 * d5';
    end
end
scovdig5 = scovdig5/(sum(labelTrain(:)==5));

[scov_eigvec_dig5 scov_eigval_dig5] = eig(scovdig5);
maxval_dig5 = max(scov_eigval_dig5(:));

component1_dig5=reshape(scov_eigvec_dig5(:,784),[28 28]);
component2_dig5=reshape(scov_eigvec_dig5(:,783),[28 28]);
component3_dig5=reshape(scov_eigvec_dig5(:,782),[28 28]);
component4_dig5=reshape(scov_eigvec_dig5(:,781),[28 28]);
component5_dig5=reshape(scov_eigvec_dig5(:,780),[28 28]);
component6_dig5=reshape(scov_eigvec_dig5(:,779),[28 28]);
component7_dig5=reshape(scov_eigvec_dig5(:,778),[28 28]);
component8_dig5=reshape(scov_eigvec_dig5(:,777),[28 28]);
component9_dig5=reshape(scov_eigvec_dig5(:,776),[28 28]);
component10_dig5=reshape(scov_eigvec_dig5(:,775),[28 28]);
figure(3)
subplot(2,5,1);
imshow(component1_dig5,[])
subplot(2,5,2);
imshow(component2_dig5,[])
subplot(2,5,3);
imshow(component3_dig5,[])
title('Principal Components')
subplot(2,5,4);
imshow(component4_dig5,[])
subplot(2,5,5);
imshow(component5_dig5,[])
subplot(2,5,6);
imshow(component6_dig5,[])
subplot(2,5,7);
imshow(component7_dig5,[])
subplot(2,5,8);
imshow(component8_dig5,[])
subplot(2,5,9);
imshow(component9_dig5,[])
subplot(2,5,10);
imshow(component10_dig5,[])
%}

%Part 2
%Part 2b
digit = 130;  %change this from 5, 10, 20, 30, 40, 60, 90, 130, 180, 250, 350
ytest = zeros(500,digit);
for testcount=1:500
    tiprime = reshape(imageTest(:,:,testcount),[784 1]) - m;   
    A=zeros(digit,784);
        for i=1:digit
            A(i,:)=scov_eigvec(:,784+1-i);
        end
        
        ytest(testcount,:)=A*tiprime;
        %ytrain=A*xprime;
    
    testcount=testcount+1;
end
ytrain = zeros(5000,digit);
for traincount=1:5000
    xprime = reshape(imageTrain(:,:,traincount),[784 1]) - m;   
    A_train=zeros(digit,784);
        for i=1:digit
            A_train(i,:)=scov_eigvec(:,784+1-i);
        end
        
        ytrain(traincount,:)=A_train*xprime;
    
    traincount=traincount+1;
end

%BDR 
%%{
totals = [sum(labelTrain(:)==0) sum(labelTrain(:)==1) sum(labelTrain(:)==2) sum(labelTrain(:)==3) sum(labelTrain(:)==4) sum(labelTrain(:)==5) sum(labelTrain(:)==6) sum(labelTrain(:)==7) sum(labelTrain(:)==8) sum(labelTrain(:)==9)];
%Create Means Here
class0 = zeros(1,digit);class1 = zeros(1,digit);class2 = zeros(1,digit);class3 = zeros(1,digit);class4 = zeros(1,digit);class5 = zeros(1,digit);class6 = zeros(1,digit);class7 = zeros(1,digit);class8 = zeros(1,digit);class9 = zeros(1,digit);
for i=1:5000    
    if labelTrain(i) == 0
        class0=class0+ytrain(i,:);
    elseif labelTrain(i) == 1
        class1=class1+ytrain(i,:);           
    elseif labelTrain(i)==2
        class2=class2+ytrain(i,:);
    elseif labelTrain(i)==3
        class3=class3+ytrain(i,:);
    elseif labelTrain(i)==4
        class4=class4+ytrain(i,:);
    elseif labelTrain(i)==5
        class5=class5+ytrain(i,:);
    elseif labelTrain(i)==6
        class6=class6+ytrain(i,:);
    elseif labelTrain(i)==7
        class7=class7+ytrain(i,:);
    elseif labelTrain(i)==8
        class8=class8+ytrain(i,:);
    elseif labelTrain(i)==9
        class9=class9+ytrain(i,:);
    end
    i=i+1;
end
class0=class0/totals(1);class1=class1/totals(2);class2=class2/totals(3);class3=class3/totals(4);class4=class4/totals(5);class5=class5/totals(6);class6=class6/totals(7);class7=class7/totals(8);class8=class8/totals(9);class9=class9/totals(10);
catclass = cat(3,class0,class1,class2,class3,class4,class5,class6,class7,class8,class9);

        



%%{
istarmatrix = zeros(1,10);
for j=1:500    
    for i=1:10
        subtraction = (ytest(j,:)-catclass(:,:,i));
      
        
        %%{
                        for jcov = 0:9
                        count = 0;
                            b = zeros(digit,digit);
                           for icov = 1:5000
                               if labelTrain(icov) == jcov
                                    diff = ytrain(icov,:) - catclass(:,:,jcov+1);
                                    b = b + diff'*diff;
                                    count = count+1;
                               end
                           end
                    trainCov5(:,:,jcov+1) = b/count;
               end
        %}
        
        
        %E = cov(ytest(j,:)',catclass(:,:,i)');
        E=trainCov5(:,:,i);
        istarmatrix(i) = (-0.5*((subtraction(:)')/(E)*subtraction(:))-(1/2)*((digit*log10(2*pi)))+(log10(1/10)));
        i=i+1;
    end
    [argvalue istar(j)] = max(istarmatrix(:));
    istar(j) = istar(j)-1; %normalize values between 0 and 9 instead of 1 and 10
    j=j+1;
end
istar=istar.';

%%{
misses = zeros(1, 10);
hits = zeros(1, 10);
for x=1:500    
    if labelTest(x) == istar(x)
        if labelTest(x)==0
            hits(1) = hits(1)+1;
        elseif labelTest(x)==1
            hits(2) = hits(2) + 1;
                    elseif labelTest(x)==2
            hits(3) = hits(3)+1;
                    elseif labelTest(x)==3
            hits(4) = hits(4)+1;
                    elseif labelTest(x)==4
            hits(5) = hits(5)+1;
                    elseif labelTest(x)==5
            hits(6) = hits(6)+1;
                    elseif labelTest(x)==6
            hits(7) = hits(7)+1;
                    elseif labelTest(x)==7
            hits(8) = hits(8)+1;
                    elseif labelTest(x)==8
            hits(9) = hits(9)+1;
                    elseif labelTest(x)==9
            hits(10)=hits(10)+1;
        end
    end
    if labelTest(x) ~= istar(x)
        if labelTest(x)==0
            misses(1) = misses(1)+1;
        elseif labelTest(x)==1
            misses(2) = misses(2)+1;
                    elseif labelTest(x)==2
            misses(3) = misses(3)+1;
                    elseif labelTest(x)==3
            misses(4) = misses(4)+1;
                    elseif labelTest(x)==4
            misses(5) = misses(5)+1;
                    elseif labelTest(x)==5
            misses(6) = misses(6)+1;
                    elseif labelTest(x)==6
            misses(7) = misses(7)+1;
                    elseif labelTest(x)==7
            misses(8) = misses(8)+1;
                    elseif labelTest(x)==8
            misses(9) = misses(9)+1;
                    elseif labelTest(x)==9
            misses(10) = misses(10)+1;
        end
    end
    
    x=x+1;
end

sumtotalrate = hits+misses;
errorrates=(misses./sumtotalrate).';

errors=zeros(500,1);
for a=1:500
    if istar(a) == labelTest(a)
        errors(a,1) = 0;
    else
        errors(a,1) = 1;
    end
    a=a+1;
end
sumerror = sum(errors); %sumerror is 47; 47 total errors out of 500
P_error = sumerror/500; %P_error is equal to .0940, or 9.4%
%{
image(1)
scatter([1:10],errorrates);
title('Error Rates For Classes 0-9')
xlabel('Class + 1')
ylabel('Error Rate Ratio')
%}
%}


