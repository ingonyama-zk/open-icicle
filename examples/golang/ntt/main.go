package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	runtime "github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12381"

	bls12381Ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12381/ntt"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254"

	bn254Ntt "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254/ntt"

	bls12381Fft "github.com/consensys/gnark-crypto/ecc/bls12-381/fr/fft"
	bn254Fft "github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
)

func main() {
	var logSize int
	var deviceType string

	flag.IntVar(&logSize, "s", 20, "Log size")
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

	size := 1 << logSize

	fmt.Printf("---------------------- NTT size 2^%d=%d ------------------------\n", logSize, size)

	print("Generating BN254 scalars ... ")
	startTime := time.Now()
	scalarsBn254 := bn254.GenerateScalars(size)
	println(time.Since(startTime).String())

	cfgBn254 := bn254Ntt.GetDefaultNttConfig()
	cfgBn254.IsAsync = true

	print("Generating BLS12_381 scalars ... ")
	startTime = time.Now()
	scalarsBls12381 := bls12381.GenerateScalars(size)
	println(time.Since(startTime).String())

	cfgBls12381 := bls12381Ntt.GetDefaultNttConfig()
	cfgBls12381.IsAsync = true
	cfgInitDomainBls := core.GetDefaultNTTInitDomainConfig()

	rouMontBn254, _ := bn254Fft.Generator(uint64(size))
	rouBn254 := rouMontBn254.Bits()
	rouIcicleBn254 := bn254.ScalarField{}
	limbsBn254 := core.ConvertUint64ArrToUint32Arr(rouBn254[:])
	rouIcicleBn254.FromLimbs(limbsBn254)
	bn254Ntt.InitDomain(rouIcicleBn254, cfgInitDomainBls)

	rouMontBls12381, _ := bls12381Fft.Generator(uint64(size))
	rouBls12381 := rouMontBls12381.Bits()
	rouIcicleBls12381 := bls12381.ScalarField{}
	limbsBls12381 := core.ConvertUint64ArrToUint32Arr(rouBls12381[:])
	rouIcicleBls12381.FromLimbs(limbsBls12381)
	bls12381Ntt.InitDomain(rouIcicleBls12381, cfgInitDomainBls)

	print("Configuring bn254 NTT ... ")
	startTime = time.Now()

	streamBn254, _ := runtime.CreateStream()

	cfgBn254.StreamHandle = streamBn254

	var nttResultBn254 core.DeviceSlice

	_, e := nttResultBn254.MallocAsync(scalarsBn254.SizeOfElement(), size, streamBn254)
	if e != runtime.Success {
		errorString := fmt.Sprint(
			"Bn254 Malloc failed: ", e)
		panic(errorString)
	}

	println(time.Since(startTime).String())

	print("Configuring Bls12381 NTT ... ")
	startTime = time.Now()

	streamBls12381, _ := runtime.CreateStream()

	cfgBls12381.StreamHandle = streamBls12381

	var nttResultBls12381 core.DeviceSlice

	_, e = nttResultBls12381.MallocAsync(scalarsBls12381.SizeOfElement(), size, streamBls12381)
	if e != runtime.Success {
		errorString := fmt.Sprint(
			"Bls12_381 Malloc failed: ", e)
		panic(errorString)
	}

	println(time.Since(startTime).String())

	print("Executing bn254 NTT on device ... ")
	startTime = time.Now()

	err := bn254Ntt.Ntt(scalarsBn254, core.KForward, &cfgBn254, nttResultBn254)
	if err != runtime.Success {
		errorString := fmt.Sprint(
			"bn254 Ntt failed: ", e)
		panic(errorString)
	}

	nttResultBn254Host := make(core.HostSlice[bn254.ScalarField], size)
	nttResultBn254Host.CopyFromDeviceAsync(&nttResultBn254, streamBn254)
	nttResultBn254.FreeAsync(streamBn254)
	runtime.SynchronizeStream(streamBn254)
	println(time.Since(startTime).String())

	print("Executing Bls12381 NTT on device ... ")
	startTime = time.Now()

	err = bls12381Ntt.Ntt(scalarsBls12381, core.KForward, &cfgBls12381, nttResultBls12381)
	if err != runtime.Success {
		errorString := fmt.Sprint(
			"bls12_381 Ntt failed: ", e)
		panic(errorString)
	}

	nttResultBls12381Host := make(core.HostSlice[bls12381.ScalarField], size)
	nttResultBls12381Host.CopyFromDeviceAsync(&nttResultBls12381, streamBls12381)
	nttResultBls12381.FreeAsync(streamBls12381)

	runtime.SynchronizeStream(streamBls12381)

	println(time.Since(startTime).String())
}
