console.log("Deepfake Detection Script Loaded.");

// Show pop-up when signed out and redirect
document.addEventListener("DOMContentLoaded", function () {
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has("signedout")) {
        alert("Signed out");
        window.location.href = "/"; // Redirect to home page
    }
});

// Plan selection button styling
document.addEventListener("DOMContentLoaded", function () {
    const planButtons = document.querySelectorAll(".plan-btn");

    planButtons.forEach(button => {
        button.addEventListener("click", function () {
            // Remove active class from all buttons
            planButtons.forEach(btn => btn.classList.remove("bg-blue-500", "text-white"));

            // Add active class to clicked button
            this.classList.add("bg-blue-500", "text-white");
        });
    });
});

// Logout functionality
document.getElementById("logoutButton")?.addEventListener("click", function () {
    fetch("/logout", { method: "POST" })
        .then(response => {
            if (response.redirected) {
                window.location.href = response.url; // Redirect to homepage
            }
        })
        .catch(error => console.error("Logout Error:", error));
});
