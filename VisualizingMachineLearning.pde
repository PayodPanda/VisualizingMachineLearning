// Author: Payod Panda
// Date: 8/24/2016
// Tested with Processing 3.0b4
// This is a stab at visualizing machine learning to see how it can help
// the understanding on some learning algorithms. To start with, I'm simply
// focusing on a logistic regression problem with variable updates using 
// gradient descent.
// note: Not very optimized in its current form. 
// note: Currently it runs over the complete dataset (not separating validation / test 
//       datasets from superset), so will experience overfitting.
//       The focus was on visualization and creating a possible GUI, not on the correctness
//       of the ML algorithm and the actual learning.
// note: Implement costs

import peasy.*;
PeasyCam cam;

Table model;

float[] precip, temperature, prediction;
int[] tillage, actual;
float precipMax, precipMin, temperatureMax, temperatureMin;
int tillageMax, tillageMin;
int count;
FloatList b0Complete, b1Complete, b2Complete, b3Complete, pComplete;
float b0, b1, b2, b3, p, alpha;
PFont font;

boolean showDifference, showModel;
IntList order;

PShape b0Path;

void setup() {
    size(displayWidth, displayHeight, P3D);
    cam = new PeasyCam(this, width/2.0, height/2.0, 0, 800);
    frameRate(30);
    ortho();
    smooth(16);
    font = createFont("Consolas", 96);
    textFont(font);

    model = loadTable("Prediction.csv", "header");    
    count = model.getRowCount();

    precip = new float[count];
    temperature = new float[count];
    prediction = new float[count];
    tillage = new int[count];
    actual = new int[count];

    precipMax = 0;
    precipMin = 9999;
    temperatureMax = 0;
    temperatureMin = 9999;
    tillageMax = 0;
    tillageMin = 9999;

    int index = 0;
    order = new IntList();

    for (TableRow row : model.rows()) {

        tillage[index] = row.getInt("Tillage");    // y
        temperature[index] = row.getFloat("AverageT");        //x
        precip[index] = row.getFloat("JulPr");        //z

        actual[index] = row.getInt("tsclero");
        prediction[index] = row.getFloat("dis");

        if (tillage[index] > tillageMax) tillageMax = tillage[index];
        if (precip[index] > precipMax) precipMax = precip[index];
        if (temperature[index] > temperatureMax) temperatureMax = temperature[index];

        if (tillage[index] < tillageMin) tillageMin = tillage[index];
        if (precip[index] < precipMax) precipMin = precip[index];
        if (temperature[index] < temperatureMin) temperatureMin = temperature[index];

        order.append(index);

        index++;
    }

    println(count);
    
    b0Complete = new FloatList();
    b1Complete = new FloatList();
    b2Complete = new FloatList();
    b3Complete = new FloatList();
    pComplete = new FloatList();
    
    b0 = 37;
    b1 = -2;
    b2 = 0;
    b3 = 0;
    p = 0;
    
    alpha = 0.08;
}


void draw() {
    background(255);
    translate(width/8, height/2, 0);
    scale(0.5);
    float boxSize = 50;
    float probabilitySize = 200;
    strokeWeight(4);
    lights();

    alpha = alpha*0.99999;
    order.shuffle();        // to shuffle the order for each iteration
    
    if(frameCount%1 == 0){
        b0Complete.append(b0);
        b1Complete.append(b1);
        b2Complete.append(b2);
        b3Complete.append(b3);
        pComplete.append(p);
    }
    
    for (int j=0; j<model.getRowCount(); j++) {
        int i = order.get(j);
        pushMatrix();
        colorMode(HSB, 1, 100, 100);
        translate(map(temperature[i], temperatureMin, temperatureMax, -width/4, width/4) +  (tillage[i]-1) * width/2, 0, map(precip[i], precipMin, precipMax, -height/2, height/2));    
        //map(tillage[i], tillageMin, tillageMax, (probabilitySize) * (tillageMax-tillageMin), -(probabilitySize) * (tillageMax-tillageMin))
        if(!showModel){
            // gradient descent:
            p = 1/(1+ exp(-(b0 + b1*temperature[i] + /*b2*tillage[i] + */b3*precip[i] )));
            b0 = b0 + alpha * (actual[i] - p) * (1-p) * p * 1;
            b1 = b1 + alpha * (actual[i] - p) * (1-p) * p * temperature[i];
            //b2 = b2 + alpha * (actual[i] - p) * (1-p) * p * tillage[i];
            b3 = b3 + alpha * (actual[i] - p) * (1-p) * p * precip[i];
        } else {
            p = prediction[i];
        }

        if (!showDifference) {
            fill(0.5169 - p, 100, 100);
            noStroke();
            translate(0, -(p*probabilitySize)/2, 0);
            box(boxSize, -(p*probabilitySize), boxSize);

            if (actual[i] == 1) {
                noFill();
                stroke(0);
                translate(0, -((actual[i] - p)*probabilitySize)/2, 0);        
                box(boxSize, -(actual[i]*probabilitySize), boxSize);
            }
        } else {
            fill(0.5169 - p, 100, 100);
            noStroke();
            if (abs(actual[i]-p) > 0.5) {
                stroke(0, 100, 100);
            }
            translate(0, -((actual[i]-p)*probabilitySize)/2, 0);
            box(boxSize, -((actual[i]-p)*probabilitySize), boxSize);
        }
        colorMode(RGB, 255, 255, 255);
        popMatrix();
    }
    cam.beginHUD();
    {
        textSize(16);
        fill(0);
        text("alpha: " + nfs(alpha, 2, 6) + "\nepoch:  " + frameCount, 20, 100);
        color b0Color, b1Color, b2Color, b3Color;
        b0Color = color(0, 255, 255);
        b1Color = color(255, 0, 255);
        b2Color = color(255, 255, 255);
        b3Color = color(255, 255, 0);
        int heightPadding = 60, widthPadding = 160;
        int vizHeight = height-(heightPadding*2);
        float b0height = 0, b1height = 0, b2height = 0, b3height = 0;
        int vizSize = 5;
        
        int totalCount = pComplete.size();
        int viewCount = (totalCount>width-widthPadding)?(width-widthPadding):totalCount;
        
        FloatList b0Temp, b1Temp, b2Temp, b3Temp;    
        b0Temp = new FloatList();
        b1Temp = new FloatList();
        b2Temp = new FloatList();
        b3Temp = new FloatList();
        
        // below i represents the index in the bigger total index (b0Complete)
        for(int i=totalCount-viewCount; i < totalCount; i++){
            // here append the value of b0Complete.get(i) to an empty FloatList that you declare before this for loop
            // this will give you the interesting subset of the complete history that you want to visualize on screen
            b0Temp.append(b0Complete.get(i));
            b1Temp.append(b1Complete.get(i));
            b2Temp.append(b2Complete.get(i));
            b3Temp.append(b3Complete.get(i));
        }
        //println(count);
        //noStroke();
        stroke(64);
        strokeWeight(0.5);
        noStroke();
        fill(64);
        rect(width-widthPadding, 0, widthPadding, height);
        for(int i=0; i < viewCount; i+=1){
            //b0Path.vertex(width-(totalCount-i), height-(map(b0Complete.get(i), b0Complete.min(), b0Complete.max(), 0, 200)));
            //println(width-(totalCount-i), height-(map(b0Complete.get(i), b0Complete.min(), b0Complete.max(), 0, 200)));
            float positionX = width-widthPadding-viewCount+i;
            b0height = height-(map(b0Temp.get(i), b0Temp.min(), b0Temp.max(), heightPadding, vizHeight)); 
            b1height = height-(map(b1Temp.get(i), b1Temp.min(), b1Temp.max(), heightPadding, vizHeight)); 
            b2height = height-(map(b2Temp.get(i), b2Temp.min(), b2Temp.max(), heightPadding, vizHeight)); 
            b3height = height-(map(b3Temp.get(i), b3Temp.min(), b3Temp.max(), heightPadding, vizHeight)); 
            fill(b0Color);
            ellipse(positionX, b0height, vizSize, vizSize);
            fill(b1Color);
            ellipse(positionX, b1height, vizSize, vizSize);
            //ellipse(positionX, b2height, vizSize, vizSize);
            fill(b3Color);
            ellipse(positionX, b3height, vizSize, vizSize);
        }
        fill(b0Color);
        text("b0: " + nfs(b0, 2, 6), width-widthPadding+10, b0height);
        fill(b1Color);
        text("b1: " + nfs(b1, 2, 6), width-widthPadding+10, b1height);
        //text("b2: " + nfs(b2, 2, 6), width-100, b2height);
        fill(b3Color);
        text("b3: " + nfs(b3, 2, 6), width-widthPadding+10, b3height);
    }
    cam.endHUD();
}

void keyPressed() {
    if (key == ' ') {
        showDifference = !showDifference;
    }
    if(key== 'p'){
        showModel = !showModel;
    }
    if (key == CODED) {
        if (keyCode == UP) {
            alpha += 0.01;
        } else {
            if (keyCode == DOWN) {
                alpha -= 0.01;
            }
        }
    }
}