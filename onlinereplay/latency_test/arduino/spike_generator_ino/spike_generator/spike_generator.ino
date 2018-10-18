/*
Generate a train of pulses that will be interpreted as spikes by the fake mouse
*/

/********** SETTINGS **********/
int StartPin = 7; // start of the time bin: a TTL on this pin
                  // will signal the real-time software to read the data for
                  // the duration of the time bin
int MousePin = 12; // to the fake mouse to simulate spikes

int PW = 25; // in microseconds (not very accurate, more than 30 in reality) 27 is good
int CORRECTION = 5; //
int BIN_DURATION = 10; // in milliseconds
int WAIT_START = 8; // in seconds
int N_SPIKES_PER_BIN = 10; // number of spikes in the replay per tetrode
double RATIO_FINAL = 0.5; // in [0, 1) the higher, the lower the added latency should be
/*******************************/


// derived params
int interspike_interval =  BIN_DURATION * 1000 / N_SPIKES_PER_BIN; // in microsec
int time_no_spikes_begin = (1 - RATIO_FINAL) * interspike_interval; // in microsec
int time_no_spikes_end = RATIO_FINAL * interspike_interval; // in microsec
int WAIT_START_MS = WAIT_START * 1000;

void setup() {
  pinMode(StartPin, OUTPUT);
  pinMode(MousePin, OUTPUT);
  randomSeed(analogRead(0));
}

void loop() {
  delay(WAIT_START_MS);
  generate_TTL(StartPin, PW);
  while (1) {
    generate_spike_train(N_SPIKES_PER_BIN, MousePin, interspike_interval, time_no_spikes_begin, time_no_spikes_end);
  }
}


/* supporting routines */
void generate_spike_train(int nspikes, int pin, int interspike_interval, int time_no_spikes_begin, int time_no_spikes_end)
{
  delayMicroseconds(time_no_spikes_begin);
  for (int i=0; i < nspikes-1; ++ i) {
      generate_TTL(pin, PW);
      delayMicroseconds(interspike_interval - PW - CORRECTION);
  }
  if (nspikes > 0) {
    generate_TTL(pin, PW);
    delayMicroseconds(time_no_spikes_end - PW - CORRECTION);
  }
}

void generate_spike_train_jitter(int nspikes, int pin, int interspike_interval, int time_no_spikes_begin, int time_no_spikes_end, double jitter_factor)
{
  int jitter_new;
  int jitter_old = random(-time_no_spikes_begin * jitter_factor, time_no_spikes_begin * jitter_factor);
  delayMicroseconds(time_no_spikes_begin + jitter_old);
  for (int i=0; i < nspikes-2; ++ i) {
      generate_TTL(pin, PW);
      jitter_new = random(-interspike_interval * jitter_factor, interspike_interval * jitter_factor);
      delayMicroseconds(interspike_interval - PW - jitter_old + jitter_new);
      jitter_old = jitter_new;
  }
  generate_TTL(pin, PW);
  delayMicroseconds(interspike_interval - PW - jitter_old);
  generate_TTL(pin, PW);
  delayMicroseconds(time_no_spikes_end - PW);
}

void generate_TTL(int pin, int pw)
{
  digitalWrite(pin, HIGH);
  delayMicroseconds(pw);
  digitalWrite(pin, LOW);
}
