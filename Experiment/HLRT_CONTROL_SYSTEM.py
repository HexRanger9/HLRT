"""
HLRT Geo-EM Amplifier Control System
====================================
Primary Controller: Raspberry Pi Pico W
Secondary Validation: Arduino Uno

Author: Ryan Tabor (Silmaril Technologies)
Version: HLRT.beta Canon v1.0
Date: January 2026
"""

# ============================================================================
# RASPBERRY PI PICO W - MAIN CONTROL CODE
# ============================================================================

PICO_MAIN_CODE = '''
from machine import Pin, PWM, ADC
import time

class Config:
    PWM_PIN = 0
    PWM_FREQ = 1000  # 1 kHz
    DEFAULT_DUTY = 0.6  # 60%
    MAX_TEMP = 60  # Celsius
    MAX_VOLTAGE = 9  # Volts
    COOLDOWN_PERIOD = 60  # seconds

class GeoEMAmplifier:
    def __init__(self):
        self.coil_pwm = PWM(Pin(Config.PWM_PIN))
        self.coil_pwm.freq(Config.PWM_FREQ)
        self.coil_pwm.duty_u16(0)
        self.is_active = False
        self.last_activation = 0
        print("Geo-EM Amplifier initialized")
        
    def activate(self, duty_cycle=0.6, duration=5.0):
        """Activate coil with specified duty cycle and duration"""
        duty_u16 = int(duty_cycle * 65535)
        print(f"ACTIVATING: {duty_cycle*100:.0f}% for {duration}s")
        
        self.is_active = True
        self.coil_pwm.duty_u16(duty_u16)
        time.sleep(duration)
        self.coil_pwm.duty_u16(0)
        self.is_active = False
        self.last_activation = time.time()
        
        print("DEACTIVATED")
        return True
        
    def emergency_stop(self):
        """Immediate shutdown"""
        self.coil_pwm.duty_u16(0)
        self.is_active = False
        print("EMERGENCY STOP")

# Main execution
if __name__ == "__main__":
    amp = GeoEMAmplifier()
    print("System ready for testing")
'''

# ============================================================================
# ARDUINO VALIDATION CODE
# ============================================================================

ARDUINO_CODE = '''
/*
 * HLRT Validation Controller - Arduino Uno
 * Secondary measurement with GPS timing
 */

const int ACS712_COIL = A0;
const int GPS_PPS = 2;
const float ACS712_SENSITIVITY = 0.185;
const float ACS712_ZERO = 2.5;

volatile bool pps_trigger = false;
float baseline_current = 0;

void setup() {
    Serial.begin(115200);
    pinMode(GPS_PPS, INPUT);
    attachInterrupt(digitalPinToInterrupt(GPS_PPS), pps_isr, RISING);
    calibrate_baseline();
    Serial.println("HLRT Validation Controller Ready");
}

void pps_isr() {
    pps_trigger = true;
}

float read_current(int pin) {
    int raw = analogRead(pin);
    float voltage = (raw / 1024.0) * 5.0;
    return (voltage - ACS712_ZERO) / ACS712_SENSITIVITY;
}

void calibrate_baseline() {
    float total = 0;
    for (int i = 0; i < 100; i++) {
        total += read_current(ACS712_COIL);
        delay(10);
    }
    baseline_current = total / 100.0;
    Serial.print("Baseline: ");
    Serial.println(baseline_current, 3);
}

void loop() {
    float current = read_current(ACS712_COIL);
    float boost = ((current - baseline_current) / baseline_current) * 100;
    
    Serial.print("I:");
    Serial.print(current, 3);
    Serial.print(" Boost:");
    Serial.print(boost, 2);
    Serial.println("%");
    
    if (boost >= 4.0) {
        Serial.println("*** SUCCESS: HLRT VALIDATED ***");
    }
    
    delay(100);
}
'''

# ============================================================================
# DATA ANALYSIS (Python - PC Side)
# ============================================================================

import json
from datetime import datetime

class HLRTDataAnalyzer:
    """Analyze experimental data from Geo-EM Amplifier tests"""
    
    TARGET_BOOST_MIN = 0.04      # 4%
    TARGET_BOOST_TARGET = 0.056  # 5.6%
    TARGET_BOOST_EXCEPTIONAL = 0.06  # 6%
    
    def __init__(self):
        self.trials = []
        self.results = {}
        
    def add_trial(self, baseline_current, peak_current, duration, notes=""):
        """Add a single trial measurement"""
        boost = (peak_current - baseline_current) / baseline_current
        
        trial = {
            'timestamp': datetime.now().isoformat(),
            'baseline_A': baseline_current,
            'peak_A': peak_current,
            'boost_fraction': boost,
            'boost_percent': boost * 100,
            'duration_s': duration,
            'notes': notes,
            'success': boost >= self.TARGET_BOOST_MIN
        }
        
        self.trials.append(trial)
        return trial
        
    def analyze_all_trials(self):
        """Statistical analysis of all trials"""
        if not self.trials:
            return None
            
        boosts = [t['boost_fraction'] for t in self.trials]
        
        import statistics
        
        self.results = {
            'trial_count': len(self.trials),
            'mean_boost': statistics.mean(boosts),
            'std_dev': statistics.stdev(boosts) if len(boosts) > 1 else 0,
            'max_boost': max(boosts),
            'min_boost': min(boosts),
            'success_count': sum(1 for t in self.trials if t['success']),
            'success_rate': sum(1 for t in self.trials if t['success']) / len(self.trials)
        }
        
        # Determine overall result
        mean = self.results['mean_boost']
        if mean >= self.TARGET_BOOST_EXCEPTIONAL:
            self.results['verdict'] = 'EXCEPTIONAL - HLRT STRONGLY VALIDATED'
        elif mean >= self.TARGET_BOOST_TARGET:
            self.results['verdict'] = 'SUCCESS - HLRT VALIDATED'
        elif mean >= self.TARGET_BOOST_MIN:
            self.results['verdict'] = 'PARTIAL - Effect detected, refinement needed'
        else:
            self.results['verdict'] = 'INCONCLUSIVE - No significant effect'
            
        return self.results
        
    def generate_report(self):
        """Generate text report of results"""
        if not self.results:
            self.analyze_all_trials()
            
        report = []
        report.append("=" * 60)
        report.append("HLRT VALIDATION EXPERIMENT REPORT")
        report.append("Silmaril Technologies")
        report.append("=" * 60)
        report.append("")
        report.append(f"Total Trials: {self.results['trial_count']}")
        report.append(f"Mean Boost: {self.results['mean_boost']*100:.2f}%")
        report.append(f"Std Dev: {self.results['std_dev']*100:.2f}%")
        report.append(f"Range: {self.results['min_boost']*100:.2f}% - {self.results['max_boost']*100:.2f}%")
        report.append(f"Success Rate: {self.results['success_rate']*100:.1f}%")
        report.append("")
        report.append(f"VERDICT: {self.results['verdict']}")
        report.append("=" * 60)
        
        return "\n".join(report)
        
    def save_to_json(self, filename):
        """Save all data to JSON file"""
        data = {
            'trials': self.trials,
            'analysis': self.results,
            'generated': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        return filename

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("HLRT Control System Code Module")
    print("=" * 40)
    print("This module contains:")
    print("  - Raspberry Pi Pico W control code")
    print("  - Arduino Uno validation code")  
    print("  - Python data analysis tools")
    print("")
    print("Deploy PICO_MAIN_CODE to Pico W")
    print("Deploy ARDUINO_CODE to Arduino Uno")
    print("Use HLRTDataAnalyzer for data processing")
    
    # Demo analysis
    analyzer = HLRTDataAnalyzer()
    
    # Example trial data (replace with actual measurements)
    analyzer.add_trial(10.67, 11.20, 5.0, "Trial 1")
    analyzer.add_trial(10.65, 11.18, 5.0, "Trial 2")
    analyzer.add_trial(10.70, 11.25, 5.0, "Trial 3")
    
    print("\n" + analyzer.generate_report())
