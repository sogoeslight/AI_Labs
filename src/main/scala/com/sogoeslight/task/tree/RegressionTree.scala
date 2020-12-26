package com.sogoeslight.task.tree

import com.sogoeslight.task.preparation.Data_Akbilgic._

import scala.collection.mutable.ArrayBuffer

object RegressionTree extends App {
  def create(): Unit = {
    // values for calculating entropy
    val avg: Double = learning.transpose.last.sum / ds.length
    val dispTotal: Double = math.sqrt(learning.transpose.last.map(elem => math.pow(elem - avg, 2)).sum / learning.length)

    def getDisp(ds: ArrayBuffer[ArrayBuffer[Double]]): Double = math.sqrt(ds.transpose.last.map(elem => math.pow(elem - avg, 2)).sum / ds.length)

    val positive: Double = convertedData.count(_.last >= dispTotal)
    val negative: Double = convertedData.length - positive
    val total: Double = convertedData.length

    def log2: Double => Double = (x: Double) => math.log10(x) / math.log10(2.0)

    def calcEntropy(pos: Double, neg: Double, total: Double): Double = (pos, neg) match {
      case (0, 0) => 0
      case (_, 0) | (0, _) => 1
      case _ => -1 * (pos / total) * log2(pos / total) - (neg / total) * log2(neg / total)
    }

    val totalEntropy: Double = calcEntropy(positive, negative, total)

    // find averages of each attribute values
    def getDelimiters(ds: ArrayBuffer[ArrayBuffer[Double]]): ArrayBuffer[Double] = {
      def findOne(ds: ArrayBuffer[ArrayBuffer[Double]]): ArrayBuffer[Double] = {
        ds.transpose.map(_.sum / convertedData.length)
      }

      findOne(ds)
    }

    def calcInfoGains(ds: ArrayBuffer[ArrayBuffer[Double]]): ArrayBuffer[BigDecimal] = {
      val delimiters: ArrayBuffer[Double] = getDelimiters(ds)
      val informationGains: ArrayBuffer[BigDecimal] = ArrayBuffer[BigDecimal]()

      def countOnes(moreOrLess: ArrayBuffer[ArrayBuffer[Double]]): Double = moreOrLess.count(p => p.last == 1.0)

      def countZeroes(moreOrLess: ArrayBuffer[ArrayBuffer[Double]]): Double = moreOrLess.count(p => p.last == 0)

      for (index <- 0 to ds(0).size - 2) {
        val moreThanDelimiter: ArrayBuffer[ArrayBuffer[Double]] = ds.filter(_ (index) >= delimiters(index))
        val attrMoreEntropy: Double = calcEntropy(countOnes(moreThanDelimiter), countZeroes(moreThanDelimiter), moreThanDelimiter.size)

        val lessThanDelimiter: ArrayBuffer[ArrayBuffer[Double]] = ds.filter(_ (index) < delimiters(index))
        val attrLessEntropy: Double = calcEntropy(countOnes(lessThanDelimiter), countZeroes(lessThanDelimiter), lessThanDelimiter.size)

        informationGains += totalEntropy - (moreThanDelimiter.size / total) * attrMoreEntropy - (lessThanDelimiter.size / total) * attrLessEntropy
      }

      informationGains
    }

    def getMaxInfoGain(IGs: ArrayBuffer[BigDecimal]): Int = IGs.indexOf(IGs.max)

    def splitNode(ds: ArrayBuffer[ArrayBuffer[Double]], bestAttributeIndex: Int, del: Double): (Node, Node) = {
      val (leftChild, rightChild) = (Node(), Node())

      for (row <- ds) {
        val removed: Double = row.remove(bestAttributeIndex)
        if (removed >= del) leftChild.dataset += row else rightChild.dataset += row
      }

      (leftChild, rightChild)
    }

    def mostCommonClass(ds: ArrayBuffer[ArrayBuffer[Double]]): String = {
      val negatives: Int = ds.count(_.last == 0)
      val positives: Int = ds.count(_.last == 1)
      if (positives > negatives) "1"
      else "0"
    }

    /**
     * Main algorithm for decision tree creating
     *
     * @param ds            - dataset
     * @param attributeList - list of attributes
     * @return Tree
     */
    def train(ds: ArrayBuffer[ArrayBuffer[Double]], attributeList: Array[String]): Node = {
      val root: Node = Node()
      val dsToTrain: ArrayBuffer[ArrayBuffer[Double]] = ds.to(ArrayBuffer).map(_.to(ArrayBuffer))
      val attributes: ArrayBuffer[String] = attributeList.to(ArrayBuffer)
      root.disp = getDisp(ds)

      val classColumn: ArrayBuffer[Double] = dsToTrain.transpose.last

      if (attributes.length <= 1) { // check if any attributes left
        root.leafAvg = classColumn.sum / ds.length
        root.label = s"${root.leafAvg} +- ${root.disp}"
      } else {
        val bestAttributeIndex = getMaxInfoGain(calcInfoGains(dsToTrain))
        root.bestAttributeIndex = Some(bestAttributeIndex)
        root.label = attributes(bestAttributeIndex)
        root.delimiter = getDelimiters(dsToTrain)(bestAttributeIndex)
        root.dataset = dsToTrain

        val (left, right) = splitNode(dsToTrain, bestAttributeIndex, root.delimiter)
        root.left = left
        root.left.dataset = left.dataset
        root.right = right
        root.right.dataset = right.dataset

        val updateAttributesList: ArrayBuffer[String] = attributes -= attributes(bestAttributeIndex)

        def newNode(node: Node, attrList: ArrayBuffer[String], isLeft: Boolean): Unit = {
          if (node.dataset.isEmpty || attrList.isEmpty) {
            node.label = mostCommonClass(node.dataset)
          } else {
            if (isLeft) root.left = train(root.left.dataset, updateAttributesList.toArray)
            else root.right = train(root.right.dataset, updateAttributesList.toArray)
          }
        }

        newNode(left, attributes, isLeft = true)
        newNode(right, attributes, isLeft = false)
      }
      root
    }

    /**
     *
     * @param tree tree for testing
     * @param ds   testing dataset
     * @return accuracy in %
     */
    def testTree(tree: Node, ds: Array[Array[Double]], isRegression: Boolean = false): Double = {
      def check(node: Node, ds: Array[Array[Double]]): Double = {
        var correctAnswersCounter: Int = 0

        if (!isRegression) {
          for (row <- ds) {
            val expected: String = row.last.toInt.toString

            var testNode: Node = node
            while (testNode.bestAttributeIndex.isDefined) {
              if (row(testNode.bestAttributeIndex.get) < testNode.delimiter) testNode = testNode.right
              else testNode = testNode.left
            }
            val realValue: String = testNode.label

            if (expected == realValue) correctAnswersCounter += 1
          }
        } else {
          for (row <- ds) {
            val expected: Double = row.last

            var testNode: Node = node
            while (testNode.bestAttributeIndex.isDefined) {
              if (row(testNode.bestAttributeIndex.get) < testNode.delimiter) testNode = testNode.right
              else testNode = testNode.left
            }

            if (expected <= tree.leafAvg + tree.disp && expected >= tree.leafAvg - tree.disp) correctAnswersCounter += 1
            }
          }

        correctAnswersCounter
      }

      check(tree, ds) / ds.length * 100
    }

    def printTreeAndAllTheData(): Unit = {
      val tree: Node = train(learning.to(ArrayBuffer).map(_.to(ArrayBuffer)), attributes)

      println("\nLAB 6 - Decision Tree\n")
      tree.draw()
      println()
      tree.printDepth()
      tree.printAmountOfLeaves()

      println("- Metrics: Information gain")
      println(f"- Accuracy for test (${test.length} rows) array: ${testTree(tree, test, isRegression = true)}%.2f%%")
      println(f"- Accuracy for learning (${learning.length} rows) array: ${testTree(tree, learning, isRegression = true)}%.2f%%")
      println("\n____________________________________________________________________________________")
    }

    printTreeAndAllTheData()
  }
}
