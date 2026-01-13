package com.example.faceauthverifier

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import coil.compose.rememberAsyncImagePainter
import com.example.faceauthverifier.ui.theme.FaceAuthVerifierTheme
import androidx.activity.compose.rememberLauncherForActivityResult
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.net.URLDecoder
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import android.util.Log



class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            FaceAuthVerifierTheme {
                val navController = rememberNavController()
                NavHost(navController, startDestination = "imagePicker") {
                    composable("imagePicker") { ImagePickerScreen(navController::navigate) }
                    composable("result/{imageUri}") { backStackEntry ->
                        val uri = backStackEntry.arguments?.getString("imageUri")?.let {
                            URLDecoder.decode(it, StandardCharsets.UTF_8.toString())
                        }
                        ResultScreen(uri)
                    }
                }
            }
        }
    }
}

@Composable
fun ImagePickerScreen(navigateToResult: (String) -> Unit) {
    val context = LocalContext.current

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val uri = result.data?.data
            uri?.let {
                try {
                    context.contentResolver.takePersistableUriPermission(
                        it,
                        Intent.FLAG_GRANT_READ_URI_PERMISSION
                    )
                } catch (e: SecurityException) {
                    e.printStackTrace()
                }

                val encoded = URLEncoder.encode(it.toString(), StandardCharsets.UTF_8.toString())
                navigateToResult("result/$encoded")
            }
        }
    }

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
        modifier = Modifier.fillMaxSize()
    ) {
        Button(onClick = {
            val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                type = "image/*"
                addCategory(Intent.CATEGORY_OPENABLE)
                flags = Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION
            }
            launcher.launch(intent)
        }) {
            Text("Выбрать изображение")
        }
    }
}

@Composable
fun ResultScreen(imageUri: String?) {
    val context = LocalContext.current
    var resultText by remember { mutableStateOf("Анализ изображения...") }

    LaunchedEffect(imageUri) {
        imageUri?.let {
            try {
                val uri = Uri.parse(it)
                val inputStream = context.contentResolver.openInputStream(uri)
                val bytes = inputStream?.readBytes()
                inputStream?.close()

                if (bytes != null) {
                    val client = OkHttpClient()
                    val requestBody = MultipartBody.Builder()
                        .setType(MultipartBody.FORM)
                        .addFormDataPart(
                            "file",
                            "image.jpg",
                            RequestBody.create("image/*".toMediaTypeOrNull(), bytes)
                        )
                        .build()

                    val request = Request.Builder()
                        .url("http://10.0.2.2:8000/predict") // сервер FastAPI
                        .post(requestBody)
                        .build()

                    val response = client.newCall(request).execute()
                    val responseText = response.body?.string() ?: "Нет ответа"
                    resultText = "Сервер: $responseText"
                } else {
                    resultText = "Ошибка: не удалось прочитать изображение"
                }
            } catch (e: Exception) {
                resultText = "Ошибка: ${e.message}"
            }
        }
    }

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text(text = resultText, style = MaterialTheme.typography.headlineMedium)
    }
}