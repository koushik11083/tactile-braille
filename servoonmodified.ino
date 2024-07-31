#include <Servo.h>
//left 70-115
//right 70-35
//wires of servo should be facing downwards towards nano

//  mys1=D3           mys2=D5
//  mys3=D6           mys4=D9
//  mys5=D10           mys6=D11
//
//          nano
//All pwm pins 3, 5, 6 , 9, 10, 11
Servo mys1;
Servo mys2;
Servo mys3;
Servo mys4;
Servo mys5;
Servo mys6;
int stop_time;
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  mys1.attach(3);
  mys2.attach(5);
  mys3.attach(6);
  mys4.attach(9);
  mys5.attach(10);
  mys6.attach(11);
  stop_time=4000;
  mys1.write(0);
  mys2.write(0);
  mys3.write(0);
  mys4.write(0);
  mys5.write(0);
  mys6.write(0);
}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available()>0)
  {
  char c = Serial.read();

  switch (c) {
    case 'A':
      mys1.write(90);
      delay(stop_time);
      mys1.write(0);
      break;
    case 'B':
      mys1.write(90);
      mys3.write(90);
      delay(stop_time);
      mys1.write(0);
      mys3.write(0);
      break;
    case 'C':
      mys1.write(90);
      mys2.write(90);
      delay(stop_time);
      mys1.write(0);
      mys2.write(0);
      break;
    case 'D':
        mys1.write(90);
        mys2.write(90);
        mys4.write(90);
        delay(stop_time);
        mys1.write(0);
        mys2.write(0);
        mys4.write(0);
      break;
    case 'E':
        mys1.write(90);
        mys4.write(90);
        delay(stop_time);
        mys1.write(0);
        mys4.write(0);
        break;
    case 'F':
      mys1.write(90);
      mys2.write(90);
      mys3.write(90);
      delay(stop_time);
      mys1.write(0);
      mys2.write(0);
      mys3.write(0);
      break;
    case 'G':
      mys1.write(90);
      mys2.write(90);
      mys3.write(90);
      mys4.write(90);;
      delay(stop_time);
      mys1.write(0);
      mys2.write(0);
      mys3.write(0);
      mys4.write(0);
      break;
    case 'H':
      mys1.write(90);
      mys3.write(90);
      mys4.write(90);
      delay(stop_time);
      mys1.write(0);
      mys3.write(0);
      mys4.write(0);
      break;
    case 'I':
      mys2.write(90);
      mys3.write(90);
      delay(stop_time);
      mys2.write(0);
      mys3.write(0);
      break;
    case 'J':
      mys2.write(90);
      mys3.write(90);
      mys4.write(90);
      delay(stop_time);
      mys2.write(0);
      mys3.write(0);
      mys4.write(0);
      break;
    case 'K':
      mys1.write(90);
      mys5.write(90);
      delay(stop_time);
      mys1.write(0);
      mys5.write(0);
      break;
    case 'L':
      mys1.write(90);
      mys3.write(90);
      mys5.write(90);
      delay(stop_time);
      mys1.write(0);
      mys3.write(0);
      mys5.write(0);
      break;
    case 'M':
      mys1.write(90);
      mys2.write(90);
      mys5.write(90);
      delay(stop_time);
      mys1.write(0);
      mys2.write(0);
      mys5.write(0);
      break;
    case 'N':
      mys1.write(90);
      mys2.write(90);
      mys4.write(90);
      mys5.write(90);
      delay(stop_time);
      mys1.write(0);
      mys2.write(0);
      mys4.write(0);
      mys5.write(0);
      break;
    case 'O':
      mys1.write(90);
      mys4.write(90);
      mys5.write(90);
      delay(stop_time);
      mys1.write(0);
      mys4.write(0);
      mys5.write(0);
      break;
    case 'P':
      mys1.write(90);
      mys2.write(90);
      mys3.write(90);
      mys5.write(90);
      delay(stop_time);
      mys1.write(0);
      mys2.write(0);
      mys3.write(0);
      mys5.write(0);
      break;
    case 'Q':
      mys1.write(90);
      mys2.write(90);
      mys3.write(90);
      mys4.write(90);
      mys5.write(90);
      delay(stop_time);
      mys1.write(0);
      mys2.write(0);
      mys3.write(0);
      mys4.write(0);
      mys5.write(0);
      break;
    case 'R':
      mys1.write(90);
      mys3.write(90);
      mys4.write(90);
      mys5.write(90);
      delay(stop_time);
      mys1.write(0);
      mys3.write(0);
      mys4.write(0);
      mys5.write(0);
      break;
    case 'S':
      mys2.write(90);
      mys3.write(90);
      mys5.write(90);
      delay(stop_time);
      mys2.write(0);
      mys3.write(0);
      mys5.write(0);
      break;
    case 'T':
      mys2.write(90);
      mys3.write(90);
      mys4.write(90);
      mys5.write(90);
      delay(stop_time);
      mys2.write(0);
      mys3.write(0);
      mys4.write(0);
      mys5.write(0);
      break;
    case 'U':
      mys1.write(90);
      mys5.write(90);
      mys6.write(90);
      delay(stop_time);
      mys1.write(0);
      mys5.write(0);
      mys6.write(0);
      break;
    case 'V':
      mys1.write(90);
      mys3.write(90);
      mys5.write(90);
      mys6.write(90);
      delay(stop_time);
      mys1.write(0);
      mys3.write(0);
      mys5.write(0);
      mys6.write(0);
      break;
    case 'W':
      mys2.write(90);
      mys3.write(90);
      mys4.write(90);
      mys6.write(90);
      delay(stop_time);
      mys2.write(0);
      mys3.write(0);
      mys4.write(0);
      mys6.write(0);
      break;
    case 'X':
      mys1.write(90);
      mys2.write(90);
      mys5.write(90);
      mys6.write(90);
      delay(stop_time);
      mys1.write(0);
      mys2.write(0);
      mys5.write(0);
      mys6.write(0);
      break;
    case 'Y':
      mys1.write(90);
      mys2.write(90);
      mys4.write(90);
      mys5.write(90);
      mys6.write(90);
      delay(stop_time);
      mys1.write(0);
      mys2.write(0);
      mys4.write(0);
      mys5.write(0);
      mys6.write(0);
      break;
    case 'Z':
      mys1.write(90);
      mys4.write(90);
      mys5.write(90);
      mys6.write(90);
      delay(stop_time);
      mys1.write(0);
      mys4.write(0);
      mys5.write(0);
      mys6.write(0);
      break;
    default:mys1.write(0);
      // Handle unexpected characters
      break;
  }

}

    delay(1000);

  }



