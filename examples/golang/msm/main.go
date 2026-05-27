package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12381"

	bls12381G2 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12381/g2"
	bls12381Msm "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12381/msm"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254"

	bn254G2 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/g2"
	bn254Msm "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/msm"
)

func main() {
	var logSizeMin int
	var logSizeMax int
	var deviceType string

	flag.IntVar(&logSizeMin, "l", 10, "Minimum log size")
	flag.IntVar(&logSizeMax, "u", 10, "Maximum log size")
	flag.StringVar(&deviceType, "device", "CUDA", "Device type")
	flag.Parse()

	if deviceType != "CPU" {
		runtime.LoadBackendFromEnvOrDefault()
	}

	device := runtime.CreateDevice(deviceType, 0)
	// NOTE: If you are only using a single device the entire time
	// 			then this is ok. If you are using multiple devices
	// 			then you should use runtime.RunOnDevice() instead.
	runtime.SetDefaultDevice(&device)

	sizeMax := 1 << logSizeMax

	print("Generating BN254 scalars ... ")
	startTime := time.Now()
	scalarsBn254Max := bn254.GenerateScalars(sizeMax)
	println(time.Since(startTime).String())

	print("Generating BN254 points ... ")
	startTime = time.Now()
	pointsBn254Max := bn254.GenerateAffinePoints(sizeMax)
	println(time.Since(startTime).String())

	print("Generating BN254 G2 points ... ")
	startTime = time.Now()
	pointsBn254G2Max := bn254G2.G2GenerateAffinePoints(sizeMax)
	println(time.Since(startTime).String())

	print("Generating BLS12_381 scalars ... ")
	startTime = time.Now()
	scalarsBls12381Max := bls12381.GenerateScalars(sizeMax)
	println(time.Since(startTime).String())

	print("Generating BLS12_381 points ... ")
	startTime = time.Now()
	pointsBls12381Max := bls12381.GenerateAffinePoints(sizeMax)
	println(time.Since(startTime).String())

	print("Generating BLS12_381 G2 points ... ")
	startTime = time.Now()
	pointsBls12381G2Max := bls12381G2.G2GenerateAffinePoints(sizeMax)
	println(time.Since(startTime).String())

	for logSize := logSizeMin; logSize <= logSizeMax; logSize++ {

		// Define the size of the problem, here 2^18.
		size := 1 << logSize

		fmt.Printf("---------------------- MSM size 2^%d=%d ------------------------\n", logSize, size)

		// println(scalarsBls12381, pointsBls12381, pointsBn254G2)
		// println(scalarsBn254, pointsBn254, pointsBls12381G2)

		print("Configuring bn254 MSM ... ")
		startTime = time.Now()

		scalarsBn254 := scalarsBn254Max[:size]
		pointsBn254 := pointsBn254Max[:size]
		pointsBn254G2 := pointsBn254G2Max[:size]

		cfgBn254 := core.GetDefaultMSMConfig()
		cfgBn254G2 := core.GetDefaultMSMConfig()
		cfgBn254.IsAsync = true
		cfgBn254G2.IsAsync = true

		streamBn254, _ := runtime.CreateStream()
		streamBn254G2, _ := runtime.CreateStream()

		cfgBn254.StreamHandle = streamBn254
		cfgBn254G2.StreamHandle = streamBn254G2

		var projectiveBn254 bn254.Projective
		var projectiveBn254G2 bn254G2.G2Projective

		var msmResultBn254 core.DeviceSlice
		var msmResultBn254G2 core.DeviceSlice

		_, e := msmResultBn254.MallocAsync(projectiveBn254.Size(), 1, streamBn254)
		if e != runtime.Success {
			errorString := fmt.Sprint(
				"Bn254 Malloc failed: ", e)
			panic(errorString)
		}
		_, e = msmResultBn254G2.MallocAsync(projectiveBn254G2.Size(), 1, streamBn254G2)
		if e != runtime.Success {
			errorString := fmt.Sprint(
				"Bn254 Malloc G2 failed: ", e)
			panic(errorString)
		}

		println(time.Since(startTime).String())

		print("Configuring Bls12381 MSM ... ")
		startTime = time.Now()

		scalarsBls12381 := scalarsBls12381Max[:size]
		pointsBls12381 := pointsBls12381Max[:size]
		pointsBls12381G2 := pointsBls12381G2Max[:size]

		cfgBls12381 := core.GetDefaultMSMConfig()
		cfgBls12381G2 := core.GetDefaultMSMConfig()
		cfgBls12381.IsAsync = true
		cfgBls12381G2.IsAsync = true

		streamBls12381, _ := runtime.CreateStream()
		streamBls12381G2, _ := runtime.CreateStream()

		cfgBls12381.StreamHandle = streamBls12381
		cfgBls12381G2.StreamHandle = streamBls12381G2

		var projectiveBls12381 bls12381.Projective
		var projectiveBls12381G2 bls12381G2.G2Projective

		var msmResultBls12381 core.DeviceSlice
		var msmResultBls12381G2 core.DeviceSlice

		_, e = msmResultBls12381.MallocAsync(projectiveBls12381.Size(), 1, streamBls12381)
		if e != runtime.Success {
			errorString := fmt.Sprint(
				"Bls12_381 Malloc failed: ", e)
			panic(errorString)
		}
		_, e = msmResultBls12381G2.MallocAsync(projectiveBls12381G2.Size(), 1, streamBls12381G2)
		if e != runtime.Success {
			errorString := fmt.Sprint(
				"Bls12_381 Malloc G2 failed: ", e)
			panic(errorString)
		}

		println(time.Since(startTime).String())

		print("Executing bn254 MSM on device ... ")
		startTime = time.Now()

		currentDevice, _ := runtime.GetActiveDevice()
		print("Device: ", currentDevice.GetDeviceType())

		e = bn254Msm.Msm(scalarsBn254, pointsBn254, &cfgBn254, msmResultBn254)
		if e != runtime.Success {
			errorString := fmt.Sprint(
				"bn254 Msm failed: ", e)
			panic(errorString)
		}
		e = bn254G2.G2Msm(scalarsBn254, pointsBn254G2, &cfgBn254G2, msmResultBn254G2)
		if e != runtime.Success {
			errorString := fmt.Sprint(
				"bn254 Msm G2 failed: ", e)
			panic(errorString)
		}

		msmResultBn254Host := make(core.HostSlice[bn254.Projective], 1)
		msmResultBn254G2Host := make(core.HostSlice[bn254G2.G2Projective], 1)

		msmResultBn254Host.CopyFromDeviceAsync(&msmResultBn254, streamBn254)
		msmResultBn254G2Host.CopyFromDeviceAsync(&msmResultBn254G2, streamBn254G2)

		msmResultBn254.FreeAsync(streamBn254)
		msmResultBn254G2.FreeAsync(streamBn254G2)

		runtime.SynchronizeStream(streamBn254)
		runtime.SynchronizeStream(streamBn254G2)

		println(time.Since(startTime).String())

		print("Executing Bls12381 MSM on device ... ")
		startTime = time.Now()

		currentDevice, _ = runtime.GetActiveDevice()
		print("Device: ", currentDevice.GetDeviceType())

		e = bls12381Msm.Msm(scalarsBls12381, pointsBls12381, &cfgBls12381, msmResultBls12381)
		if e != runtime.Success {
			errorString := fmt.Sprint(
				"bls12_381 Msm failed: ", e)
			panic(errorString)
		}
		e = bls12381G2.G2Msm(scalarsBls12381, pointsBls12381G2, &cfgBls12381G2, msmResultBls12381G2)
		if e != runtime.Success {
			errorString := fmt.Sprint(
				"bls12_381 Msm G2 failed: ", e)
			panic(errorString)
		}

		msmResultBls12381Host := make(core.HostSlice[bls12381.Projective], 1)
		msmResultBls12381G2Host := make(core.HostSlice[bls12381G2.G2Projective], 1)

		msmResultBls12381Host.CopyFromDeviceAsync(&msmResultBls12381, streamBls12381)
		msmResultBls12381G2Host.CopyFromDeviceAsync(&msmResultBls12381G2, streamBls12381G2)

		msmResultBls12381.FreeAsync(streamBls12381)
		msmResultBls12381G2.FreeAsync(streamBls12381G2)

		runtime.SynchronizeStream(streamBls12381)
		runtime.SynchronizeStream(streamBls12381G2)

		println(time.Since(startTime).String())
	}
}
