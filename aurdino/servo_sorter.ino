#include <Servo.h>

Servo myServo;

void setup() {
  Serial.begin(9600);
  myServo.attach(9);
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');

    if (cmd == "A") myServo.write(30);
    else if (cmd == "B") myServo.write(150);
    else if (cmd == "A1") myServo.write(60);
    else if (cmd == "B1") myServo.write(120);

    delay(1000);
    myServo.write(90); // neutral
  }
}
