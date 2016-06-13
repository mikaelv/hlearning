{-# LANGUAGE DataKinds #-}

module Main where

import System.Exit (exitFailure)
import Test.HUnit
import Test.Framework.Providers.API
import Test.Framework.Providers.HUnit
import Test.Framework.Runners.Console (defaultMain)
import Numeric.LinearAlgebra.Static
import qualified Numeric.LinearAlgebra as LA
import HLearning.Util

testToBinaryVector = testCase "should transform the labels vector to a binary matrix" $ actual @?= expected
  where expected = extract (matrix [0,1,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,1] :: L 2 10)
        actual = extract . toBinMatrix $ vec2 1 9

tests = testGroup "HLearning tests" [ testToBinaryVector ]

main = defaultMain [tests]
