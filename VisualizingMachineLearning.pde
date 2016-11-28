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
int[] tillage, actual, state;
float precipMax, precipMin, temperatureMax, temperatureMin;
int tillageMax, tillageMin;
int count;
FloatList b0Complete, b1Complete, b2Complete, b3Complete, pComplete;
float b0, b1, b2, b3, p, alpha;
int epoch, elapsedTime, wastedTime;
float clockIn, clockOut;
int frameSkip;
PFont font;

boolean showDifference, showModel, showBoxes, combineTillage, combineState;
IntList order;

PShape b0Path;

void setup() {
    size(displayWidth, displayHeight, P3D);
    cam = new PeasyCam(this, width/2.0, height/2.0, 0, 800);
    cam.setMinimumDistance(0.1);
    cam.setMaximumDistance(999999);
    frameRate(30);
    ortho();
    smooth(16);
    font = createFont("Consolas", 96);
    textFont(font);
    textSize(16);

    model = loadTable("Prediction.csv", "header");    
    count = model.getRowCount();

    precip = new float[count];
    temperature = new float[count];
    prediction = new float[count];
    tillage = new int[count];
    state = new int[count];
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
        state[index] = row.getInt("State");
        switch(state[index]) {
            case(17):
            state[index] = 0;
            break;
            case(19):
            state[index] = 1;
            break;
            case(27):
            state[index] = 2;
            break;
            case(39):
            state[index] = 3;
            break;
        default:
            state[index] = 0;
            break;
        }

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

    b0 = 5.71;
    b1 = -0.46;
    b2 = 0;
    b3 = 0.029;
    p = 0;

    alpha = 0.05;
    epoch = 0;
    clockIn = 0;
    elapsedTime = 0;

    frameSkip = 1;
}


void draw() {
    background(32, 16, 12);
    background(16);
    translate(width/2, height/2, 0);
    //scale(0.5);
    float boxSize = 15;
    float probabilitySize = combineTillage?400:100;
    float plotWidth = combineState?width/2:width/6;
    //float plotHeight = combineTillage?4*probabilitySize:probabilitySize;
    float plotDepth = combineTillage?height/2:height/6;
    float[] rotation = cam.getRotations();
    float[] pos = cam.getPosition();
    lights();
    strokeWeight(2);


    if (!showModel && frameCount%1 == 0) {
        b0Complete.append(b0);
        b1Complete.append(b1);
        b2Complete.append(b2);
        b3Complete.append(b3);
        pComplete.append(p);
        alpha = alpha*0.99999;
        epoch++;
        elapsedTime = int(millis()/1000) - wastedTime;
        order.shuffle();        // to shuffle the order for each iteration
    }

    cam.beginHUD();
    {
        noLights();
        if (!combineTillage) {
            for (int i=0; i<4; i++) {
                //these are the background rectangles
                colorMode(RGB, 255, 255, 255);
                fill((3-i+1)*16.0, (3-i+1)*8.0, (3-i+1)*6.0);
                noStroke();
                //rect(0, i*height/4, width, height/4);
                colorMode(HSB, 1, 100, 100, 100);
            }
        }
        lights();

        translate(width/2, height/2, 0);
        for (int j=0; j<model.getRowCount(); j++) {
            int i = order.get(j);
            pushMatrix();
            {
                colorMode(HSB, 1, 100, 100, 100);

                if (!combineTillage) {
                    translate(0, (tillage[i]-1) * height/4.0 + height/8 + height/16 - height/2 , 0);
                } else {
                    translate(0, 0, 0);
                }

                if (!combineState) {
                    translate((state[i]) * width/4.0 + width/8 - width/2, 0, 0);
                } 


                pushMatrix();
                {

                    rotateX(-rotation[0]);
                    rotateY(-rotation[1]);
                    rotateZ(-rotation[2]);
                    //translate(-pos[0], -pos[1], -pos[2]);

                    scale(1);
                    translate(map(temperature[i], temperatureMin, temperatureMax, -plotWidth/2, plotWidth/2), 0, map(precip[i], precipMin, precipMax, -plotDepth/2, plotDepth/2));

                    //map(tillage[i], tillageMin, tillageMax, (probabilitySize) * (tillageMax-tillageMin), -(probabilitySize) * (tillageMax-tillageMin))
                    if (!showModel) {
                        // gradient descent:
                        p = 1/(1+ exp(-(b0 + b1*temperature[i] + /*b2*tillage[i] + */b3*precip[i] )));
                        
                        /* 
                            formula: (intercept) + (state) + (average/tillage) + (precipitation)
                        */
                        
                        b0 = b0 + alpha * (actual[i] - p) * (1-p) * p * 1;
                        b1 = b1 + alpha * (actual[i] - p) * (1-p) * p * temperature[i];
                        //b2 = b2 + alpha * (actual[i] - p) * (1-p) * p * tillage[i];
                        b3 = b3 + alpha * (actual[i] - p) * (1-p) * p * precip[i];
                    } else {
                        p = prediction[i];
                    }

                    if (!showDifference) {
                        /*
                        fill(map(p, 0.5, 0, 0, 0.2), 100, 100, map(p, 0, 0.5, 20, 100));
                        noStroke();
                         */
                        strokeWeight(map(p, 0, 0.5, 0.5, 2));
                        stroke(map(p, 0.5, 0, 0, 0.2), 100, 100, map(p, 0, 0.5, 4, 100));
                        noFill();
                        translate(0, -(p*probabilitySize)/2, 0);
                        box(boxSize, -(p*probabilitySize), boxSize);

                        if (actual[i] == 1 && showBoxes) {
                            noFill();
                            //stroke(0, 0, 25);
                            translate(0, (p*probabilitySize)/2, 0);   
                            beginShape(QUAD_STRIP);
                            {
                                fill(0, 100, 100, 32);
                                vertex(-boxSize/2, -(actual[i]*probabilitySize), -boxSize/2);
                                vertex(-boxSize/2, -(actual[i]*probabilitySize), boxSize/2);
                                vertex(boxSize/2, -(actual[i]*probabilitySize), -boxSize/2);
                                vertex(boxSize/2, -(actual[i]*probabilitySize), boxSize/2);
                                fill(0.2, 100, 100, 32);
                                vertex(boxSize/2, 0, -boxSize/2);
                                vertex(boxSize/2, 0, boxSize/2);
                                vertex(-boxSize/2, 0, -boxSize/2);
                                vertex(-boxSize/2, 0, boxSize/2);
                                fill(0, 100, 100, 32);
                                vertex(-boxSize/2, -(actual[i]*probabilitySize), -boxSize/2);
                                vertex(-boxSize/2, -(actual[i]*probabilitySize), boxSize/2);
                            }
                            endShape(CLOSE);   
                            beginShape(QUAD_STRIP);
                            {
                                fill(0, 100, 100, 32);
                                vertex(-boxSize/2, -(actual[i]*probabilitySize), boxSize/2);
                                vertex(boxSize/2, -(actual[i]*probabilitySize), boxSize/2);
                                fill(0.2, 100, 100, 32);
                                vertex(-boxSize/2, 0, boxSize/2);
                                vertex(boxSize/2, 0, boxSize/2);
                            }
                            endShape(CLOSE);
                            beginShape(QUAD_STRIP);
                            {
                                fill(0, 100, 100, 32);
                                vertex(-boxSize/2, -(actual[i]*probabilitySize), -boxSize/2);
                                vertex(boxSize/2, -(actual[i]*probabilitySize), -boxSize/2);
                                fill(0.2, 100, 100, 32);
                                vertex(-boxSize/2, 0, -boxSize/2);
                                vertex(boxSize/2, 0, -boxSize/2);
                            }
                            endShape(CLOSE);
                            //box(boxSize, -(actual[i]*probabilitySize), boxSize);
                        } else {
                            noFill();
                            stroke(0, 0, 25);
                            //translate(0, -((actual[i] - p)*probabilitySize)/2, 0);  
                            pushMatrix();
                            rotateX(PI/2);
                            //rect(-boxSize/2, -boxSize/2, boxSize, boxSize);
                            popMatrix();
                        }
                    } else {
                        /*
                        fill(map(p, 0.5, 0, 0, 0.2), 100, 100, map(p, 0, 0.5, 20, 100));
                         noStroke();
                         */
                        strokeWeight(map(p, 0, 0.5, 0.5, 2));
                        stroke(map(p, 0.5, 0, 0, 0.2), 100, 100, map(p, 0, 0.5, 4, 100));
                        noFill();
                        if (abs(actual[i]-p) > 0.99 && showBoxes) {
                            stroke(0, 100, 100);
                            strokeWeight(2);
                        }
                        translate(0, -((actual[i]-p)*probabilitySize)/2, 0);
                        box(boxSize, -((actual[i]-p)*probabilitySize), boxSize);
                    }
                    colorMode(RGB, 255, 255, 255);
                }
                popMatrix();
            }
            popMatrix();
        }
    }
    cam.endHUD();

    cam.beginHUD();
    {

        noLights();
        
        if (!showModel) {
            fill(240, 0, 0);
            textSize(48);
            text("visualizing machine learning", 32, 88); 
            pushMatrix();
            {
                fill(64);
                translate(0, 60, 0);
                textSize(16);
                if (showModel) text("Showing data from Prof. Mina's predictive model. [p] to switch to machine learning trial mode.", 32, 64); 
                else text("Showing data from the machine learning trial. [p] to switch to Prof. Mina's model.", 32, 64); 
                if (showDifference) text("Showing the difference between observed value and prediction. [spacebar] to switch to probability.", 32, 88); 
                else text("Showing the probability of finding disease. [spacebar] to switch to deviation.", 32, 88);
                if (showBoxes) text("The translucent boxes represent observations of the disease in data. [b] to turn off.", 32, 110); 
                else text("[b] to show disease observations.", 32, 110);
                if (combineTillage) text("Showing the data with tillage combined. [c] to classify.", 32, 132); 
                else text("Showing the data classified by tillage. [c] to combine.", 32, 132);
                textAlign(RIGHT);
                if (!showModel) text("learning rate (alpha): " + nfs(alpha, 2, 6) + "\niteration (epoch):  " + epoch + "\ntime elapsed:" + elapsedTime + "s", width-32, 64);
                textAlign(LEFT);
            }
            popMatrix();
        }
        

        if (!showModel) {
            color b0Color, b1Color, b2Color, b3Color;
            b0Color = color(0, 255, 255);
            b1Color = color(255, 0, 255);
            b2Color = color(255, 255, 255);
            b3Color = color(255, 255, 0);

            int heightPaddingT = 30, // top
                heightPaddingB = 30, // bottom
                heightPadding = heightPaddingT + heightPaddingB, // total
                vizHeight = (height/2)-(heightPadding*2), // height
                widthPaddingR = 160, // right
                widthPaddingL = 160, // left
                widthPadding = widthPaddingR + widthPaddingL;                       // total

            float b0height = 0, b1height = 0, b2height = 0, b3height = 0;
            int vizSize = 2;

            int totalCount = pComplete.size();
            int viewCount = (totalCount>width-widthPadding)?(width-widthPadding):totalCount;

            FloatList b0Temp, b1Temp, b2Temp, b3Temp;    
            b0Temp = new FloatList();
            b1Temp = new FloatList();
            b2Temp = new FloatList();
            b3Temp = new FloatList();

            // below, i represents the index in the bigger total index (b0Complete)
            for (int i=totalCount-viewCount; i < totalCount; i+=1) {
                // here append the value of b0Complete.get(i) to an empty FloatList that you declare before this for loop
                // this will give you the interesting subset of the complete history that you want to visualize on screen
                b0Temp.append(b0Complete.get(i));
                b1Temp.append(b1Complete.get(i));
                b2Temp.append(b2Complete.get(i));
                b3Temp.append(b3Complete.get(i));
            }

            PShape b0Viz = createShape();
            b0Viz.beginShape();
            b0Viz.noFill();
            b0Viz.stroke(b0Color);
            b0Viz.strokeWeight(vizSize);

            PShape b1Viz = createShape();
            b1Viz.beginShape();
            b1Viz.noFill();
            b1Viz.stroke(b1Color);
            b1Viz.strokeWeight(vizSize);
            /*
            PShape b2Viz = createShape();
             b2Viz.beginShape();
             b2Viz.noFill();
             b2Viz.stroke(b2Color);
             b2Viz.strokeWeight(vizSize);
             */
            PShape b3Viz = createShape();
            b3Viz.beginShape();
            b3Viz.noFill();
            b3Viz.stroke(b3Color);
            b3Viz.strokeWeight(vizSize);
            
            // viewCount is the number of elements you show at any given time, generally equal to the width of the plot
            for (int i=0; i < viewCount; i+=1) {
                float positionX = width-widthPaddingR-viewCount+i;

                // in the local dataset that we're interested in, has there been any change at all? If yes, then use that to determine height of point, else place it in the center of the chart.
                b0height = (b0Temp.min() != b0Temp.max()) ? height-(map(b0Temp.get(i), b0Temp.min(), b0Temp.max(), heightPadding, vizHeight)) : height-(vizHeight/2); 
                b1height = (b1Temp.min() != b1Temp.max()) ? height-(map(b1Temp.get(i), b1Temp.min(), b1Temp.max(), heightPadding, vizHeight)) : height-(vizHeight/2); 
                b2height = (b2Temp.min() != b2Temp.max()) ? height-(map(b2Temp.get(i), b2Temp.min(), b2Temp.max(), heightPadding, vizHeight)) : height-(vizHeight/2); 
                b3height = (b3Temp.min() != b3Temp.max()) ? height-(map(b3Temp.get(i), b3Temp.min(), b3Temp.max(), heightPadding, vizHeight)) : height-(vizHeight/2); 
                b0Viz.vertex(positionX, b0height);
                b1Viz.vertex(positionX, b1height);
                //b2Viz.vertex(positionX, b2height);
                b3Viz.vertex(positionX, b3height);
            }
            b3Viz.endShape();
            //b2Viz.endShape();
            b1Viz.endShape();
            b0Viz.endShape();
            shape(b0Viz);
            shape(b1Viz);
            //shape(b2Viz);
            shape(b3Viz);

            // the container lines
            colorMode(RGB, 255, 255, 255);
            stroke(64);
            strokeWeight(2);
            line(widthPaddingL, height - vizHeight - heightPadding, widthPaddingL, height);
            line(width-widthPaddingR, height - vizHeight - heightPadding, width-widthPaddingR, height);
            //noStroke();

            // the text
            if (abs(b0height-b1height) < 16) {
                b0height = b1height-16;
            }
            if (abs(b1height-b3height) < 16) {
                b3height = b1height+16;
            }
            if (abs(b0height-b3height) < 16) {
                b3height = b0height+16;
            }
            if (totalCount>0) {
                fill(b0Color);
                text("b0: " + nfs(b0Complete.get(totalCount-1), 2, 6), width-widthPaddingR+10, b0height);
                fill(b1Color);
                text("b1: " + nfs(b1Complete.get(totalCount-1), 2, 6), width-widthPaddingR+10, b1height);
                //text("b2: " + nfs(b2, 2, 6), width-100, b2height);
                fill(b3Color);
                text("b3: " + nfs(b3Complete.get(totalCount-1), 2, 6), width-widthPaddingR+10, b3height);
            }
        }
    }
    cam.endHUD();
}

void keyPressed() {
    if (key == ' ') {
        showDifference = !showDifference;
    }
    if (key== 'p') {
        if (showModel) {
            clockIn = int(millis()/1000);
            wastedTime += clockIn - clockOut;
        }
        if (!showModel) clockOut = int(millis()/1000);
        showModel = !showModel;
    }
    if (key== 'b') {
        showBoxes = !showBoxes;
    }
    if (key== 't') {
        combineTillage = !combineTillage;
    }
    if (key== 's') {
        combineState = !combineState;
    }
    if (key == CODED) {
        if (keyCode == UP) {
            alpha += 0.01;
        } 
        if (keyCode == DOWN) {
            alpha -= 0.01;
        }
        if (keyCode == LEFT) {
            frameSkip -= 10;
        }
        if (keyCode == RIGHT) {
            frameSkip += 10;
        }
    }
}