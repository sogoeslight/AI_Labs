package com.sogoeslight.task.preparation

import com.sogoeslight.task.preparation.Data.datasetRegex

import scala.collection.mutable.ArrayBuffer
import scala.io.{BufferedSource, Source}
import scala.util.Random

object Data_Akbilgic extends App {
  final lazy val path_to_data: String = "./src/main/resources/data_akbilgic.csv" // just xlsx saved as csv

  final lazy val dataset: BufferedSource = Source.fromFile(path_to_data)
  final lazy val initialData: ArrayBuffer[ArrayBuffer[String]] = dataset.getLines()
    .map(_.split(datasetRegex).to(ArrayBuffer)).to(ArrayBuffer)
  dataset.close()

  final lazy val cleanDataset: Array[Array[String]] = {
    var dataset: ArrayBuffer[ArrayBuffer[String]] = ArrayBuffer[ArrayBuffer[String]]()

    for (row <- initialData) { // removing extra spaces and empty fields
      val cleanRow: ArrayBuffer[String] = row.map(_.trim).filter(_.nonEmpty)
      if (cleanRow.size == initialData.head.length) dataset += cleanRow
    }

    dataset.toArray.map(_.toArray)
  }

  // drop dates
  final lazy val ds: Array[Array[String]] = cleanDataset.transpose.drop(1).transpose

  final lazy val attributes: Array[String] = ds.head.dropRight(1)

  // drop attributes row, convert data to double
  final lazy val convertedData: Array[Array[Double]] = ds.drop(1).map(_.map(_.toDouble))
  //final lazy val convertedData: Array[Array[Double]] = convertedData1.map(_.map(_ * 1000))

  final lazy val shuffledSamples: Array[Array[Double]] = Random.shuffle(convertedData.to(ArrayBuffer).map(_.map(identity))).toArray
  final lazy val (learning, test) = shuffledSamples.splitAt(math.ceil(convertedData.length * 0.75).toInt)
  final lazy val avg: Double = convertedData.transpose.last.sum / convertedData.length

  def prepare(): Unit = {
    println("\nLAB 6")
    println(s"\n- Rows count: ${cleanDataset.length}")
    println(s"- Attribute count: ${attributes.length}")

    println(s"\n- Learning samples: ${learning.length}")
    println(s"- Test samples: ${test.length}")
    println()
  }

  prepare()

}
