# Passive Sonar Sound Localization

This is a project I am doing in my free time at work to study how sonar and other sound localization technologies work. Currently, I can find out where a sound is originated only by azimuth.

I am working on creating a microphone array to achieve better precision and better estimate of true location. I am experimenting with recordings from an in line 4 mic array to start with, but I have a total of 8 microphone that I can use so I will eventually create a proper 2 dimensional array to give an attitude direction and a distance estimate.

To achieve this, I am using beamforming algorithms which works by shifting the microphone recordings by a certain delay until the sound power reaches its peak. This works because each microphone is spaced away from eachother by the same distance, so a sound will reach each microphones at different intervals. By looking at which microphone were delayed and by how much, we can then get an estimate of the location of the sound. 



## Development Checklist

- [X]  Find azimuth from stereo microphone recording using GCC PHAT
- [ ]  Find azimuth from a 4 microphone line array using SRP PHAT
- [ ]  Build a 2 dimenional mic array 
- [ ]  Find azimuth and attitude of sound and diplay it nicely


