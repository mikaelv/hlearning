{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ImplicitParams, ScopedTypeVariables #-}

module Main where

import System.Exit (exitFailure)
import Test.HUnit
import Test.HUnit.Approx
import Test.Framework.Providers.API
import Test.Framework.Providers.HUnit
import Test.Framework.Runners.Console (defaultMain)
import Numeric.LinearAlgebra.Static
import qualified Numeric.LinearAlgebra as LA
import HLearning.Util
import HLearning.GradientDescent

testToBinaryVector = testCase "should transform the labels vector to a binary matrix" $ actual @?= expected
  where expected = extract (matrix [0,1,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,1] :: L 2 10)
        actual = extract . toBinMatrix $ vec2 1 9


testGradientDescent = testCase "should find the minimum of x^2 + 3*x" $ let ?epsilon = 0.0001 in actual @~? expected
  where function x = x^2 + 3*x
        expected = -3/2
        tolerance = 1e-9
        stopCond = stopCondition function tolerance
        alpha = 0.1
        initValue = 12
        actual = unArg $ gradientDescent function (stopCondition function tolerance) alpha (Arg initValue)

tests = testGroup "HLearning tests" [ testToBinaryVector, testGradientDescent ]

main = defaultMain [tests]

assertFloatEqual text a b =
  assertEqual text (take 4 (show a)) (take 4 (show b))
